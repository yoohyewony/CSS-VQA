import json
import os
import pickle
import time
from os.path import join

import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import random
import copy

#import wandb

def compute_score_with_logits(logits, labels):
    logits = torch.argmax(logits, 1)
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores.sum(1)


def train(models, train_loader, eval_loader, args, qid2type):
    dataset=args.dataset
    num_epochs=args.epochs
    mode=args.mode
    run_eval=args.eval_each_epoch
    output=args.output
    #optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    total_step = 0
    best_eval_score = 0
    
    model_num = args.model_num
    
    optimizers = []
    for i in range(model_num):
        optim = torch.optim.Adamax(models[i].parameters())
        optimizers.append(optim)
    
    '''
    losses = []
    for i in range(model_num):
        losses.append()
    '''

        
    if mode=='q_debias':
        topq=args.topq
        keep_qtype=args.keep_qtype
    elif mode=='v_debias':
        topv=args.topv
        top_hint=args.top_hint
    elif mode=='q_v_debias':
        topv=args.topv
        top_hint=args.top_hint
        topq=args.topq
        keep_qtype=args.keep_qtype
        qvp=args.qvp



    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0

        t = time.time()
        for i, (v, q, a, b, hintscore,type_mask,notype_mask,q_mask) in tqdm(enumerate(train_loader), ncols=100,
                                                   desc="Epoch %d" % (epoch + 1), total=len(train_loader)):

            total_step += 1


            #########################################
            v = Variable(v).cuda().requires_grad_()
            q = Variable(q).cuda()
            q_mask=Variable(q_mask).cuda()
            a = Variable(a).cuda()
            b = Variable(b).cuda()
            hintscore = Variable(hintscore).cuda()
            type_mask=Variable(type_mask).float().cuda()
            notype_mask=Variable(notype_mask).float().cuda()
            #########################################

            loss_sum = 0
            
            if mode=='updn':
                for i in range(model_num):
                    pred, loss, _ = models[i](v, q, a, b, None)
                    if(i==0):
                        score = compute_score_with_logits(pred, a.data)
                        #print(score.sum())
                    else:
                        tmp = compute_score_with_logits(pred, a.data)
                        #print(tmp.sum())
                        score = torch.logical_or(tmp, score)
                        #print(score.shape)
                    '''if (i==0):
                        preds = torch.unsqueeze(pred, 0)    # [model_num, 512, 2274]
                    else:
                        preds = torch.cat((preds, torch.unsqueeze(pred, 0)))
                    '''
                                          
                    if (loss != loss).any():
                        raise ValueError("NaN loss")

                    loss_sum += loss
                    total_loss += loss.item()*q.size(0)    # q.size(0) = 423
                    
                    #wandb.log({"Train loss " : loss.item()*q.size(0)/512})
                    #print(loss)
                    #print(total_loss)
                    
                loss_sum.backward()
                for i in range(model_num):
                    nn.utils.clip_grad_norm_(models[i].parameters(), 0.25)
                    optimizers[i].step()
                    optimizers[i].zero_grad()

                '''for optim in optimizers:
                    optim.step()
                    optim.zero_grad()
                '''

                #total_loss += loss.item() * q.size(0)
                #batch_score = compute_score_with_logits(preds, a.data).sum()
                batch_score = score.sum()
                train_score += batch_score

            elif mode=='q_debias':
                if keep_qtype==True:
                    sen_mask=type_mask
                else:
                    sen_mask=notype_mask
                ## first train
                pred, loss, word_emb = models(v, q, a, b, None)

                word_grad = torch.autograd.grad((pred * (a > 0).float()).sum(), word_emb, create_graph=True)[0]

                if (loss != loss).any():
                    raise ValueError("NaN loss")
                loss.backward()
                nn.utils.clip_grad_norm_(models.parameters(), 0.25)                
                optim.step()
                optim.zero_grad()

                total_loss += loss.item() * q.size(0)
                batch_score = compute_score_with_logits(pred, a.data).sum()
                train_score += batch_score

                ## second train

                word_grad_cam = word_grad.sum(2)
                # word_grad_cam_sigmoid = torch.sigmoid(word_grad_cam * 1000)
                word_grad_cam_sigmoid = torch.exp(word_grad_cam * sen_mask)
                word_grad_cam_sigmoid = word_grad_cam_sigmoid * sen_mask

                w_ind = word_grad_cam_sigmoid.sort(1, descending=True)[1][:, :topq]

                q2 = copy.deepcopy(q_mask)

                m1 = copy.deepcopy(sen_mask)  ##[0,0,0...0,1,1,1,1]
                m1.scatter_(1, w_ind, 0)  ##[0,0,0...0,0,1,1,0]
                m2 = 1 - m1  ##[1,1,1...1,1,0,0,1]
                if dataset=='cpv1':
                    m3=m1*18330
                else:
                    m3 = m1 * 18455  ##[0,0,0...0,0,18455,18455,0]
                q2 = q2 * m2.long() + m3.long()

                pred, _, _ = models(v, q2, None, b, None)

                pred_ind = torch.argsort(pred, 1, descending=True)[:, :5]
                false_ans = torch.ones(pred.shape[0], pred.shape[1]).cuda()
                false_ans.scatter_(1, pred_ind, 0)
                a2 = a * false_ans
                q3 = copy.deepcopy(q)
                if dataset=='cpv1':
                    q3.scatter_(1, w_ind, 18330)
                else:
                    q3.scatter_(1, w_ind, 18455)

                ## third train

                pred, loss, _ = models(v, q3, a2, b, None)

                if (loss != loss).any():
                    raise ValueError("NaN loss")
                loss.backward()
                nn.utils.clip_grad_norm_(models.parameters(), 0.25)
                optim.step()
                optim.zero_grad()

                total_loss += loss.item() * q.size(0)

            elif mode=='v_debias':
                ## first train
                pred, loss, _ = models(v, q, a, b, None)
                visual_grad=torch.autograd.grad((pred * (a > 0).float()).sum(), v, create_graph=True)[0]

                if (loss != loss).any():
                    raise ValueError("NaN loss")
                loss.backward()
                nn.utils.clip_grad_norm_(models.parameters(), 0.25)
                optim.step()
                optim.zero_grad()

                total_loss += loss.item() * q.size(0)
                batch_score = compute_score_with_logits(pred, a.data).sum()
                train_score += batch_score

                ##second train
                v_mask = torch.zeros(v.shape[0], 36).cuda()
                visual_grad_cam = visual_grad.sum(2)
                hint_sort, hint_ind = hintscore.sort(1, descending=True)
                v_ind = hint_ind[:, :top_hint]
                v_grad = visual_grad_cam.gather(1, v_ind)

                if topv==-1:
                    v_grad_score,v_grad_ind=v_grad.sort(1,descending=True)
                    v_grad_score=nn.functional.softmax(v_grad_score*10,dim=1)
                    v_grad_sum=torch.cumsum(v_grad_score,dim=1)
                    v_grad_mask=(v_grad_sum<=0.65).long()
                    v_grad_mask[:,0] = 1
                    v_mask_ind=v_grad_mask*v_ind
                    for x in range(a.shape[0]):
                        num=len(torch.nonzero(v_grad_mask[x]))
                        v_mask[x].scatter_(0,v_mask_ind[x,:num],1)
                else:
                    v_grad_ind = v_grad.sort(1, descending=True)[1][:, :topv]
                    v_star = v_ind.gather(1, v_grad_ind)
                    v_mask.scatter_(1, v_star, 1)


                pred, _, _ = models(v, q, None, b, v_mask)

                pred_ind = torch.argsort(pred, 1, descending=True)[:, :5]
                false_ans = torch.ones(pred.shape[0], pred.shape[1]).cuda()
                false_ans.scatter_(1, pred_ind, 0)
                a2 = a * false_ans

                v_mask = 1 - v_mask

                pred, loss, _ = models(v, q, a2, b, v_mask)

                if (loss != loss).any():
                    raise ValueError("NaN loss")
                loss.backward()
                nn.utils.clip_grad_norm_(models.parameters(), 0.25)
                optim.step()
                optim.zero_grad()

                total_loss += loss.item() * q.size(0)

            elif mode=='q_v_debias':
                random_num = random.randint(1, 10)
                if keep_qtype == True:
                    sen_mask = type_mask
                else:
                    sen_mask = notype_mask
                if random_num<=qvp:
                    ## first train
                    pred, loss, word_emb = models(v, q, a, b, None)
                    word_grad = torch.autograd.grad((pred * (a > 0).float()).sum(), word_emb, create_graph=True)[0]

                    if (loss != loss).any():
                        raise ValueError("NaN loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(models.parameters(), 0.25)
                    optim.step()
                    optim.zero_grad()

                    total_loss += loss.item() * q.size(0)
                    batch_score = compute_score_with_logits(pred, a.data).sum()
                    train_score += batch_score

                    ## second train

                    word_grad_cam = word_grad.sum(2)
                    # word_grad_cam_sigmoid = torch.sigmoid(word_grad_cam * 1000)
                    word_grad_cam_sigmoid = torch.exp(word_grad_cam * sen_mask)
                    word_grad_cam_sigmoid = word_grad_cam_sigmoid * sen_mask
                    w_ind = word_grad_cam_sigmoid.sort(1, descending=True)[1][:, :topq]

                    q2 = copy.deepcopy(q_mask)

                    m1 = copy.deepcopy(sen_mask)  ##[0,0,0...0,1,1,1,1]
                    m1.scatter_(1, w_ind, 0)  ##[0,0,0...0,0,1,1,0]
                    m2 = 1 - m1  ##[1,1,1...1,1,0,0,1]
                    if dataset=='cpv1':
                        m3=m1*18330
                    else:
                        m3 = m1 * 18455  ##[0,0,0...0,0,18455,18455,0]
                    q2 = q2 * m2.long() + m3.long()

                    pred, _, _ = models(v, q2, None, b, None)

                    pred_ind = torch.argsort(pred, 1, descending=True)[:, :5]
                    false_ans = torch.ones(pred.shape[0], pred.shape[1]).cuda()
                    false_ans.scatter_(1, pred_ind, 0)
                    a2 = a * false_ans
                    q3 = copy.deepcopy(q)
                    if dataset=='cpv1':
                        q3.scatter_(1, w_ind, 18330)
                    else:
                        q3.scatter_(1, w_ind, 18455)

                    ## third train

                    pred, loss, _ = models(v, q3, a2, b, None)

                    if (loss != loss).any():
                        raise ValueError("NaN loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(models.parameters(), 0.25)
                    optim.step()
                    optim.zero_grad()

                    total_loss += loss.item() * q.size(0)


                else:
                    ## first train
                    pred, loss, _ = models(v, q, a, b, None)
                    visual_grad = torch.autograd.grad((pred * (a > 0).float()).sum(), v, create_graph=True)[0]

                    if (loss != loss).any():
                        raise ValueError("NaN loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(models.parameters(), 0.25)
                    optim.step()
                    optim.zero_grad()

                    total_loss += loss.item() * q.size(0)
                    batch_score = compute_score_with_logits(pred, a.data).sum()
                    train_score += batch_score

                    ##second train
                    v_mask = torch.zeros(v.shape[0], 36).cuda()
                    visual_grad_cam = visual_grad.sum(2)
                    hint_sort, hint_ind = hintscore.sort(1, descending=True)
                    v_ind = hint_ind[:, :top_hint]
                    v_grad = visual_grad_cam.gather(1, v_ind)

                    if topv == -1:
                        v_grad_score, v_grad_ind = v_grad.sort(1, descending=True)
                        v_grad_score = nn.functional.softmax(v_grad_score * 10, dim=1)
                        v_grad_sum = torch.cumsum(v_grad_score, dim=1)
                        v_grad_mask = (v_grad_sum <= 0.65).long()
                        v_grad_mask[:,0] = 1
                        v_mask_ind = v_grad_mask * v_ind
                        for x in range(a.shape[0]):
                            num = len(torch.nonzero(v_grad_mask[x]))
                            v_mask[x].scatter_(0, v_mask_ind[x,:num], 1)
                    else:
                        v_grad_ind = v_grad.sort(1, descending=True)[1][:, :topv]
                        v_star = v_ind.gather(1, v_grad_ind)
                        v_mask.scatter_(1, v_star, 1)

                    pred, _, _ = models(v, q, None, b, v_mask)
                    pred_ind = torch.argsort(pred, 1, descending=True)[:, :5]
                    false_ans = torch.ones(pred.shape[0], pred.shape[1]).cuda()
                    false_ans.scatter_(1, pred_ind, 0)
                    a2 = a * false_ans

                    v_mask = 1 - v_mask

                    pred, loss, _ = models(v, q, a2, b, v_mask)

                    if (loss != loss).any():
                        raise ValueError("NaN loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(models.parameters(), 0.25)
                    optim.step()
                    optim.zero_grad()

                    total_loss += loss.item() * q.size(0)

        if mode=='updn':
            total_loss /= len(train_loader.dataset)
        else:
            total_loss /= len(train_loader.dataset) * 2
        train_score = 100 * train_score / len(train_loader.dataset)

        if run_eval:
            for model in models:
                model.train(False)
                model.eval()
            results = evaluate(models, eval_loader, qid2type, model_num)
            results["epoch"] = epoch + 1
            results["step"] = total_step
            results["train_loss"] = total_loss
            results["train_score"] = train_score

            for model in models:
                model.train(True)

            eval_score = results["score"]
            bound = results["upper_bound"]
            yn = results['score_yesno']
            other = results['score_other']
            num = results['score_number']
            accuracy = results['model_score']

        logger.write('epoch %d, time: %.2f' % (epoch, time.time() - t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))

        if run_eval:
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
            logger.write('\tyn score: %.2f other score: %.2f num score: %.2f' % (100 * yn, 100 * other, 100 * num))
            for i in range(model_num):
                logger.write('\tmodel{0} eval score: {1}'.format(i, 100*accuracy[i]))

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            model_state = []
            for m in models:
                model_state.append(m.state_dict())
            state = { 'model' : model_state }
            torch.save(state, model_path)
            #for i in range(model_num):
            #    torch.save({'model{}'.format(i), models[i].state_dict(), model_path})
            best_eval_score = eval_score


def evaluate(models, dataloader, qid2type, model_num):
    score = 0
    upper_bound = 0
    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_number = 0
    total_other = 0
    accuracy = 0

    scores = []
    for v, q, a, b, qids, _ in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        v = Variable(v, requires_grad=False).cuda()
        q = Variable(q, requires_grad=False).cuda()
        for i in range(model_num):
            pred, _, _ = models[i](v, q, a, b, None)
            if(i==0):
                logit = compute_score_with_logits(pred, a.cuda()).cpu()
                scores = compute_score_with_logits(pred, a.cuda()).cpu().sum().view(1)
            else:
                tmp = compute_score_with_logits(pred, a.cuda()).cpu()
                scores = torch.cat([scores, tmp.sum().view(1)])
                logit = torch.logical_or(tmp, logit)
            '''if (i==0):
                preds = torch.unsqueeze(pred, 0)    # [model_num, 512, 2274]
            else:
                preds = torch.cat((preds, torch.unsqueeze(pred, 0)))
                    '''
        #pred, _, _ = model(v, q, None, None, None)
        #batch_score = compute_score_with_logits(preds, a.data).sum()
        #train_score += batch_score
        
        #batch_score = logit.numpy().sum(1)
        accuracy += scores
        batch_score = logit.numpy()
        score += batch_score.sum()
        upper_bound += (a.max(1)[0]).sum()
        qids = qids.detach().cpu().int().numpy()
        for j in range(len(qids)):
            qid = qids[j]
            typ = qid2type[str(qid)]
            if typ == 'yes/no':
                score_yesno += batch_score[j]
                total_yesno += 1
            elif typ == 'other':
                score_other += batch_score[j]
                total_other += 1
            elif typ == 'number':
                score_number += batch_score[j]
                total_number += 1
            else:
                print('Hahahahahahahahahahaha')

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    score_yesno /= total_yesno
    score_other /= total_other
    score_number /= total_number
    for i in range(model_num):
        accuracy[i] = accuracy[i]/len(dataloader.dataset)
    #accruacy = accuracy/len(dataloader.dataset)
    #print(len(dataloader.dataset))    219928

    results = dict(
        score=score,
        upper_bound=upper_bound,
        score_yesno=score_yesno,
        score_other=score_other,
        score_number=score_number,
        model_score = accuracy
    )
    return results
