# 事件关系分类模块
# 作者：宋杨
from nl2tensor import *
from process_control import *
import os
import re
from utils.argutils import print_args
# from pathlib import Path
import argparse
import json
import glob
import torch.optim
import numpy as np
from tqdm import trange
from language_model.transformers.configuration_electra import ElectraConfig
from my_optimizers import Ranger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from language_model.transformers import ElectraTokenizer
from nn.embeddings import ElectraModel
from models.rnn_cnn_model import *
import datetime


# 设置全局变量
def set_args(filename):
    parser = argparse.ArgumentParser()
    # 可调参数
    parser.add_argument("--train_epochs",
                        default=20,  # 默认5
                        type=int,
                        help="训练次数大小")
    parser.add_argument("--embeddings_lr",
                        default=5e-4,
                        type=float,
                        help="Embeddings初始学习步长")
    parser.add_argument("--encoder_lr",
                        default=5e-4,
                        type=float)
    parser.add_argument("--learning_rate",
                        default=5e-4,
                        type=float)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--train_batch_size",
                        default=16,  # 默认8
                        type=int,
                        help="训练时batch大小")
    parser.add_argument("--max_sent_len",
                        default=128,  # 默认256
                        type=int,
                        help="文本最大长度")
    parser.add_argument("--test_size",
                        default=.0,
                        type=float,
                        help="验证集大小")
    parser.add_argument("--train_data_filename",
                        default='data/rel_data/test.csv',
                        type=str,
                        help="The input data filename. Should contain the .csv files (or other data files) for the "
                             "task.")
    parser.add_argument("--test_data_filename",
                        default='data/rel_data/test.csv',
                        type=str)
    parser.add_argument("--train_data_dir",
                        default='data/rel_data/',
                        type=str,
                        help="The input data dir.")
    parser.add_argument("--test_data_dir",
                        default='data/rel_data/',
                        type=str)
    parser.add_argument("--mymodel_config_dir",
                        default='config/relation_classify_config.json',
                        type=str)
    parser.add_argument("--mymodel_save_dir",
                        default='checkpoint/relation_classify/',
                        type=str)
    parser.add_argument("--pretrained_model_dir",
                        default='pretrained_model/pytorch_electra_180g_large/',
                        type=str)
    parser.add_argument("--vocab_dir",
                        default='pretrained_model/pytorch_electra_180g_large/vocab.txt',
                        type=str,
                        help="The vocab data dir.")
    parser.add_argument("--rel2label",
                        default={'Causal': 0, 'Follow': 1, 'Accompany': 2, 'Concurrency': 3, 'Other': 4},
                        type=dict)
    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="训练模式")
    parser.add_argument("--do_eval",
                        default=True,
                        action='store_true',
                        help="验证模式")
    parser.add_argument("--no_gpu",
                        default=False,
                        action='store_true',
                        help="用不用gpu")
    parser.add_argument("--seed",
                        default=6,
                        type=int,
                        help="初始化时的随机数种子")
    parser.add_argument("--gradient_accumulation_steps",
                        default=1,
                        type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--optimize_on_cpu",
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU.")
    parser.add_argument("--fp16",
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit.")
    parser.add_argument("--loss_scale",
                        default=128,
                        type=float,
                        help="Loss scaling, positive power of 2 values can improve fp16 convergence.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true')
    args = parser.parse_args()
    print_args(args, parser)
    with open(filename, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    return args


# 设置全局环境
try:
    args = set_args('config/relation_classify_args.txt')
except FileNotFoundError:
    args = set_args('config/relation_classify_args.txt')
logger = get_logger()
set_environ()
today = datetime.datetime.now()
my_time = str(today.year)+'-'+str(today.month)+'-'+str(today.day)
if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    torch.distributed.init_process_group(backend='nccl')
loss_device = torch.device("cpu")
logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
    device, n_gpu, bool(args.local_rank != -1), args.fp16))
if args.gradient_accumulation_steps < 1:
    raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
        args.gradient_accumulation_steps))


# 定义一个计算准确率的函数
def accuracy(preds, labels, seq_len):
    count, right = 0.1, 0.1
    for pred, label in zip(preds, labels):
        for i in range(seq_len):
            if label[i] != len(args.tag_to_ix) - 1 and label[i] != len(args.tag_to_ix) - 2 \
                    and label[i] != len(args.tag_to_ix) - 3 and label[i] != len(args.tag_to_ix) - 4:
                count += 1
                _, p = pred[i].topk(1)
                if int(label[i]) == p[0].item():
                    right += 1
    return right / count


# 关系转label
def rel2label(t_label, args):
    try:
        return args.rel2label[t_label]
    except:
        return len(args.rel2label)-1


# 网络训练
def mymodel_train(args, logger, train_dataloader, validation_dataloader):
    config = ElectraConfig.from_pretrained(args.mymodel_config_dir)
    embedding = ElectraModel(config=config)
    model = RelClassifyModel(config=config, args=args)
    # try:
    #     output_model_file = os.path.join(args.mymodel_save_dir, 'embedding/')
    #     model_state_dict = torch.load(os.path.join(output_model_file, 'pytorch_model.bin'))
    #     embedding.load_state_dict(model_state_dict)
    # except OSError:
    #     embedding.from_pretrained(os.path.join(args.pretrained_model_dir, 'pytorch_model.bin'), config=config)
    #     print("PretrainedEmbeddingNotFound")
    # try:
    #     model.load(os.path.join(args.mymodel_save_dir, "mymodel.bin"))
    # except OSError:
    #     print("PretrainedMyModelNotFound")
    embedding.from_pretrained(os.path.join(args.pretrained_model_dir, 'pytorch_model.bin'), config=config)
    if args.fp16:
        embedding.half()
        model.half()
    embedding.to(device)
    model.to(device)
    param_optimizer1 = list(embedding.named_parameters())
    param_optimizer2 = list(model.named_parameters())
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in param_optimizer1 if any(nd in n for nd in ['embeddings'])],
         'weight_decay_rate': args.weight_decay,
         'lr': args.embeddings_lr},
        {'params': [p for n, p in param_optimizer1 if not any(nd in n for nd in ['embeddings'])],
         'lr': args.encoder_lr},
    ]
    optimizer_grouped_parameters2 = [
        {'params': [p for n, p in param_optimizer2 if any(nd in n for nd in ['encoder'])],
         'lr': args.encoder_lr},
        {'params': [p for n, p in param_optimizer2 if not any(nd in n for nd in ['encoder'])],
         'lr': args.learning_rate},
    ]
    optimizer1 = Ranger(optimizer_grouped_parameters1)
    optimizer2 = Ranger(optimizer_grouped_parameters2)
    epochs = args.train_epochs
    bio_records, train_loss_set, acc_records= [], [], []
    embedding.train()
    model.train()
    for _ in trange(epochs, desc='Epochs'):
        tr_loss = 0
        eval_loss, eval_accuracy = 0, 0
        nb_tr_steps = 0
        nb_eval_steps = 0
        tmp_loss = []
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids1, b_input_ids2, b_labels = batch
            b_input_ids1 = b_input_ids1.squeeze(1).long()
            b_input_ids2 = b_input_ids2.squeeze(1).long()
            text_embedding1 = embedding(input_ids=b_input_ids1)
            text_embedding2 = embedding(input_ids=b_input_ids2,
                                        token_type_ids=torch.ones(b_input_ids2.size(),
                                                                  dtype=torch.long, device=device))
            loss, tmp_eval_accuracy = model(text_embedding1, text_embedding2, b_labels)
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            torch.cuda.empty_cache()
            tr_loss += loss.item()
            nb_tr_steps += 1
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
            tmp_loss.append(loss.item())
        adjust_learning_rate(optimizer1, 0.9)
        adjust_learning_rate(optimizer2, 0.9)
        try:
            train_loss_set.append(tr_loss / nb_tr_steps)
            logger.info('mymodel训练损失:{:.2f},准确率为：{:.2f}%'
                        .format(tr_loss / nb_tr_steps, 100 * eval_accuracy / nb_eval_steps))
            acc_records.append(eval_accuracy / nb_eval_steps)
            bio_records.append(np.mean(train_loss_set))
        except ZeroDivisionError:
            logger.info("错误！请降低batch大小")
        embedding_to_save = embedding.module if hasattr(embedding, 'module') else embedding
        torch.save(embedding_to_save.state_dict(),
                   os.path.join(os.path.join(args.mymodel_save_dir, 'embedding/'), my_time+'pytorch_model.bin'))
        model.save(os.path.join(args.mymodel_save_dir, my_time+"mymodel.bin"))
    return embedding, model


# 网络测试
def mymodel_test(logger, test_dataloader, the_time=my_time):
    config = ElectraConfig.from_pretrained(args.mymodel_config_dir)
    embedding = ElectraModel(config=config)
    model = RelClassifyModel(config=config, args=args)
    output_model_file = os.path.join(args.mymodel_save_dir, 'embedding/')
    model_state_dict = torch.load(os.path.join(output_model_file, the_time+'pytorch_model.bin'))
    embedding.load_state_dict(model_state_dict)
    model.load(os.path.join(args.mymodel_save_dir, the_time+"mymodel.bin"))
    if args.fp16:
        embedding.half()
        model.half()
    embedding.to(device)
    model.to(device)
    embedding.eval()
    model.eval()
    acc_records = []
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps = 0
    for step, batch in enumerate(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids1, b_input_ids2, b_labels = batch
        b_input_ids1 = b_input_ids1.squeeze(1).long()
        b_input_ids2 = b_input_ids2.squeeze(1).long()
        with torch.no_grad():
            text_embedding1 = embedding(input_ids=b_input_ids1)
            text_embedding2 = embedding(input_ids=b_input_ids2,
                                        token_type_ids=torch.ones(b_input_ids2.size(),
                                                                  dtype=torch.long, device=device))
            tmp_eval_accuracy = model.test(text_embedding1, text_embedding2, b_labels)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
    try:
        logger.info('准确率为：{:.2f}%'
                    .format(100 * eval_accuracy / nb_eval_steps))
        acc_records.append(eval_accuracy / nb_eval_steps)
    except ZeroDivisionError:
        logger.info("错误！请降低batch大小")
    return acc_records


def mymodel_cal(logger, test_dataloader, the_time=my_time):
    config = ElectraConfig.from_pretrained(args.mymodel_config_dir)
    embedding = ElectraModel(config=config)
    model = RelClassifyModel(config=config, args=args)
    output_model_file = os.path.join(args.mymodel_save_dir, 'embedding/')
    model_state_dict = torch.load(os.path.join(output_model_file, the_time+'pytorch_model.bin'))
    embedding.load_state_dict(model_state_dict)
    output_model_file = os.path.join(args.mymodel_save_dir, the_time+"mymodel.bin")
    model_state_dict = torch.load(output_model_file)
    model.load_state_dict(model_state_dict)
    if args.fp16:
        embedding.half()
        model.half()
    embedding.to(device)
    model.to(device)
    embedding.eval()
    model.eval()
    target_size = len(args.rel2label)
    result = np.zeros([target_size, target_size])
    for step, batch in enumerate(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids1, b_input_ids2, b_labels = batch
        b_input_ids1 = b_input_ids1.squeeze(1).long()
        b_input_ids2 = b_input_ids2.squeeze(1).long()
        with torch.no_grad():
            text_embedding1 = embedding(input_ids=b_input_ids1)
            text_embedding2 = embedding(input_ids=b_input_ids2,
                                        token_type_ids=torch.ones(b_input_ids2.size(),
                                                                  dtype=torch.long, device=device))
            pred = model.get_guess(text_embedding1, text_embedding2)
        size = pred.size()[0]
        for i in range(size):
            try:
                result[b_labels[i], label_from_output(pred[i])] += 1
            except:
                continue
    print(result)
    return result


# 获取数据集
def get_dataloader(filenames):
    tokenizer = ElectraTokenizer.from_pretrained(args.vocab_dir)
    all_train_filenames = glob.glob(filenames + "*.csv")
    input_ids1 = []
    input_ids2 = []
    labels = []
    cnt = 0
    Text1 = re.compile('.+\$')
    Text2 = re.compile('\$.+@')
    Text3 = re.compile('@.+')
    for idx, filename in enumerate(all_train_filenames):
        for line in read_lines(filename):
            line = re.sub(u'\t', '', line)
            text1 = re.sub(u'\$', '', Text1.findall(line)[0])
            text2 = re.sub(u'@', '', re.sub(u'\$', '', Text2.findall(line)[0]))
            t_label = re.sub(u'@', '', Text3.findall(line)[0])
            tmp1, _, _ = text2ids(tokenizer, text1, args.max_sent_len)
            tmp2, _, _ = text2ids(tokenizer, text2, args.max_sent_len)
            label = rel2label(t_label, args)
            input_ids1.append(tmp1)
            input_ids2.append(tmp2)
            labels.append(label)
            cnt += 1
    train_input1, validation_input1, train_input2, validation_input2, train_labels, validation_labels = \
        train_test_split(input_ids1, input_ids2, labels, random_state=args.seed, test_size=args.test_size)

    # 将训练集tensor并生成dataloader
    train_inputs1 = torch.Tensor(train_input1)
    train_inputs2 = torch.Tensor(train_input2)
    train_labels = torch.LongTensor(train_labels)
    batch_size = args.train_batch_size
    train_data = TensorDataset(train_inputs1, train_inputs2, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                  sampler=train_sampler,
                                  batch_size=batch_size)

    if args.test_size > 0:
        # 将验证集tensor并生成dataloader
        validation_inputs1 = torch.Tensor(validation_input1)
        validation_inputs2 = torch.Tensor(validation_input2)
        validation_labels = torch.LongTensor(validation_labels)
        validation_data = TensorDataset(validation_inputs1, validation_inputs2, validation_labels)
        validation_sampler = RandomSampler(validation_data)
        validation_dataloader = DataLoader(validation_data,
                                           sampler=validation_sampler,
                                           batch_size=batch_size)
        return train_dataloader, validation_dataloader
    else:
        return train_dataloader, _

def main():
    train_dataloader, validation_dataloader = get_dataloader(args.train_data_dir)
    embedding, model = mymodel_train(args, logger, train_dataloader, validation_dataloader)
    test_dataloader, _ = get_dataloader(args.test_data_dir)
    acc_records = mymodel_test(logger, test_dataloader)
    result = mymodel_cal(logger, test_dataloader)


if __name__ == "__main__":
    main()
