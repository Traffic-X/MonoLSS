import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from lib.helpers.decode_helper import _transpose_and_gather_feat
from lib.losses.focal_loss import focal_loss_cornernet as focal_loss
from lib.losses.uncertainty_loss import laplacian_aleatoric_uncertainty_loss,laplacian_aleatoric_uncertainty_loss_new


class Hierarchical_Task_Learning:
    def __init__(self,epoch0_loss,stat_epoch_nums=5):
        self.index2term = [*epoch0_loss.keys()]
        self.term2index = {term:self.index2term.index(term) for term in self.index2term}  #term2index
        self.stat_epoch_nums = stat_epoch_nums
        self.past_losses=[]
        self.loss_graph = {'seg_loss':[],
                           'size2d_loss':[], 
                           'offset2d_loss':[],
                           'offset3d_loss':['size2d_loss','offset2d_loss'],
                           'size3d_loss':['size2d_loss','offset2d_loss'], 
                           'heading_loss':['size2d_loss','offset2d_loss'], 
                           'depth_loss':['size2d_loss','size3d_loss','offset2d_loss'],
                           }
                            


    def compute_weight(self,current_loss,epoch):
        T=140
        #compute initial weights
        loss_weights = {}
        eval_loss_input = torch.cat([_.unsqueeze(0) for _ in current_loss.values()]).unsqueeze(0)
        for term in self.loss_graph:
            if len(self.loss_graph[term])==0:
                loss_weights[term] = torch.tensor(1.0).to(current_loss[term].device)
            else:
                loss_weights[term] = torch.tensor(0.0).to(current_loss[term].device) 
        #update losses list
        if len(self.past_losses)==self.stat_epoch_nums:
            past_loss = torch.cat(self.past_losses)
            mean_diff = (past_loss[:-2]-past_loss[2:]).mean(0)
            if not hasattr(self, 'init_diff'):
                self.init_diff = mean_diff
            c_weights = 1-(mean_diff/self.init_diff).relu().unsqueeze(0)
            
            time_value = min(((epoch-5)/(T-5)),1.0)
            for current_topic in self.loss_graph:
                if len(self.loss_graph[current_topic])!=0:
                    control_weight = 1.0
                    for pre_topic in self.loss_graph[current_topic]:
                        control_weight *= c_weights[0][self.term2index[pre_topic]]      
                    loss_weights[current_topic] = time_value**(1-control_weight)
                    if loss_weights[current_topic] != loss_weights[current_topic]:
                        for pre_topic in self.loss_graph[current_topic]:
                            print('NAN===============', time_value, control_weight, c_weights[0][self.term2index[pre_topic]], pre_topic, self.term2index[pre_topic])
            #pop first list
            self.past_losses.pop(0)
        self.past_losses.append(eval_loss_input)

        return loss_weights
    def update_e0(self,eval_loss):
        self.epoch0_loss = torch.cat([_.unsqueeze(0) for _ in eval_loss.values()]).unsqueeze(0)


class LSS_Loss(nn.Module):
    def __init__(self,epoch):
        super().__init__()
        self.stat = {}
        self.epoch = epoch

    def forward(self, preds, targets):

        if targets['mask_2d'].sum() == 0:
            bbox2d_loss = 0
            bbox3d_loss = 0
            self.stat['offset2d_loss'] = 0
            self.stat['size2d_loss'] = 0
            self.stat['depth_loss'] = 0
            self.stat['offset3d_loss'] = 0
            self.stat['size3d_loss'] = 0
            self.stat['heading_loss'] = 0
        else:
            bbox2d_loss = self.compute_bbox2d_loss(preds, targets)
            bbox3d_loss = self.compute_bbox3d_loss(preds, targets)

        seg_loss = self.compute_segmentation_loss(preds, targets)

        mean_loss = seg_loss + bbox2d_loss + bbox3d_loss
        return float(mean_loss), self.stat


    def compute_segmentation_loss(self, input, target):
        input['heatmap'] = torch.clamp(input['heatmap'].sigmoid_(), min=1e-4, max=1 - 1e-4)
        loss = focal_loss(input['heatmap'], target['heatmap'])
        self.stat['seg_loss'] = loss
        return loss


    def compute_bbox2d_loss(self, input, target):
        # compute size2d loss
        
        size2d_input = extract_input_from_tensor(input['size_2d'], target['indices'], target['mask_2d'])
        size2d_target = extract_target_from_tensor(target['size_2d'], target['mask_2d'])
        size2d_loss = F.l1_loss(size2d_input, size2d_target, reduction='mean')
        # compute offset2d loss
        offset2d_input = extract_input_from_tensor(input['offset_2d'], target['indices'], target['mask_2d'])
        offset2d_target = extract_target_from_tensor(target['offset_2d'], target['mask_2d'])
        offset2d_loss = F.l1_loss(offset2d_input, offset2d_target, reduction='mean')


        loss = offset2d_loss + size2d_loss   
        self.stat['offset2d_loss'] = offset2d_loss
        self.stat['size2d_loss'] = size2d_loss
        return loss


    def compute_bbox3d_loss(self, input, target, mask_type = 'mask_2d'):
        vis_depth = input['vis_depth'][input['train_tag']]
        vis_depth_target = extract_target_from_tensor(target['vis_depth'], target[mask_type])
        
        attention_map = input['attention_map'][input['train_tag']]
        depth_mask_target_map = attention_map

        vis_depth_uncer = input['vis_depth_uncer'][input['train_tag']]
        vis_depth_loss = laplacian_aleatoric_uncertainty_loss_new(vis_depth,
                                                        vis_depth_target,
                                                        vis_depth_uncer)  
       	device = vis_depth.device
        if self.epoch < 100:
            vis_depth_loss = torch.mean(vis_depth_loss)
        else:
            attention_map_mask = attention_map.view(-1,49)
            if self.epoch<0:
                attention_map_mask = gumbel_softmax_topk(attention_map_mask, tau=1, hard=True)
            else:
                attention_map_mask = gumbel_softmax_topk(attention_map_mask,tau=1,hard=True,soft_ = True)
            attention_map_mask = attention_map_mask.view(-1,7,7)
            vis_depth_loss = torch.mean(vis_depth_loss*attention_map_mask) * 10
         
        depth_loss = vis_depth_loss * 10

        # compute offset3d loss
        offset3d_input = input['offset_3d'][input['train_tag']]
        offset3d_target = extract_target_from_tensor(target['offset_3d'], target[mask_type])
        offset3d_loss = F.l1_loss(offset3d_input, offset3d_target, reduction='mean')


        # compute size3d loss
        size3d_input = input['size_3d'][input['train_tag']]
        size3d_target = extract_target_from_tensor(target['size_3d'], target[mask_type])
        size3d_loss = F.l1_loss(size3d_input, size3d_target, reduction='mean')

        # compute heading loss
        heading_loss = compute_heading_loss(input['heading'][input['train_tag']] ,
                                            target[mask_type],  ## NOTE
                                            target['heading_bin'],
                                            target['heading_res'])

        loss = depth_loss + offset3d_loss + size3d_loss + heading_loss

        if depth_loss != depth_loss:
            print('badNAN----------------depth_loss', depth_loss)
            print(vis_depth_loss)
        if offset3d_loss != offset3d_loss:
            print('badNAN----------------offset3d_loss', offset3d_loss)
        if size3d_loss != size3d_loss:
            print('badNAN----------------size3d_loss', size3d_loss)
        if heading_loss != heading_loss:
            print('badNAN----------------heading_loss', heading_loss)

        self.stat['depth_loss'] = depth_loss
        self.stat['offset3d_loss'] = offset3d_loss
        self.stat['size3d_loss'] = size3d_loss
        self.stat['heading_loss'] = heading_loss
        # self.stat['l2_loss'] = l2_loss
        
        return loss


### ======================  auxiliary functions  =======================

def extract_input_from_tensor(input, ind, mask):
    input = _transpose_and_gather_feat(input, ind)  # B*C*H*W --> B*K*C
    return input[mask]  # B*K*C --> M * C


def extract_target_from_tensor(target, mask):
    return target[mask]

#compute heading loss two stage style  

def compute_heading_loss(input, mask, target_cls, target_reg):
    mask = mask.view(-1)   # B * K  ---> (B*K)
    target_cls = target_cls.view(-1)  # B * K * 1  ---> (B*K)
    target_reg = target_reg.view(-1)  # B * K * 1  ---> (B*K)

    # classification loss
    input_cls = input[:, 0:12]
    target_cls = target_cls[mask]
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')
    
    # regression loss
    input_reg = input[:, 12:24]
    target_reg = target_reg[mask]
    cls_onehot = torch.zeros(target_cls.shape[0], 12).cuda().scatter_(dim=1, index=target_cls.view(-1, 1), value=1)
    input_reg = torch.sum(input_reg * cls_onehot, 1)
    reg_loss = F.l1_loss(input_reg, target_reg, reduction='mean')
    
    return cls_loss + reg_loss    
'''    

def compute_heading_loss(input, ind, mask, target_cls, target_reg):
    """
    Args:
        input: features, shaped in B * C * H * W
        ind: positions in feature maps, shaped in B * 50
        mask: tags for valid samples, shaped in B * 50
        target_cls: cls anns, shaped in B * 50 * 1
        target_reg: reg anns, shaped in B * 50 * 1
    Returns:
    """
    input = _transpose_and_gather_feat(input, ind)   # B * C * H * W ---> B * K * C
    input = input.view(-1, 24)  # B * K * C  ---> (B*K) * C
    mask = mask.view(-1)   # B * K  ---> (B*K)
    target_cls = target_cls.view(-1)  # B * K * 1  ---> (B*K)
    target_reg = target_reg.view(-1)  # B * K * 1  ---> (B*K)

    # classification loss
    input_cls = input[:, 0:12]
    input_cls, target_cls = input_cls[mask], target_cls[mask]
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')

    # regression loss
    input_reg = input[:, 12:24]
    input_reg, target_reg = input_reg[mask], target_reg[mask]
    cls_onehot = torch.zeros(target_cls.shape[0], 12).cuda().scatter_(dim=1, index=target_cls.view(-1, 1), value=1)
    input_reg = torch.sum(input_reg * cls_onehot, 1)
    reg_loss = F.l1_loss(input_reg, target_reg, reduction='mean')
    
    return cls_loss + reg_loss
'''




def gumbel_softmax_topk(logits, tau=1, hard=False, eps=1e-10, dim=-1,k=1,soft_ = False):
    # type: (Tensor, float, bool, float, int) -> Tensor
    r"""
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """

    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)
    device = y_soft.device
    if hard:
        # Straight through.
        if soft_:
            y_soft_sort = torch.sort(y_soft)[0]
            #print(y_soft_sort[0])
            y_soft_times = y_soft_sort[:,1:] / y_soft_sort[:,:-1]
            #print(y_soft_times[0])
            y_times_max,y_times_max_index = torch.max(y_soft_times,-1)
            #print(y_times_max[0])
            y_thre = torch.gather(y_soft_sort,1,y_times_max_index.view(-1,1)) 
            #print(y_thre[0])
            y_thre = torch.where(y_times_max.view(-1,1)>torch.ones_like(y_thre).to(device)*1000,y_thre,torch.zeros_like(y_thre).to(y_soft_sort.device)).view(-1,1).repeat(1,49)
            #print(y_thre[0,0])
            #import pdb; pdb.set_trace()
            y_zeros = torch.zeros_like(y_soft_sort).to(y_soft_sort.device)
            #y_min_ = torch.where(y_max>y_min_*100,y_min_,y_zeros) 
            y_hard = torch.where(y_soft >= y_thre,y_soft,y_zeros)
        else:
            index = torch.topk(y_soft,20)[1]
            y_hard_temp = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            y_hard = (y_soft > 0.01) +  y_hard_temp
        ret = (y_hard - y_soft).detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


if __name__ == '__main__':
    input_cls  = torch.zeros(2, 50, 12)  # B * 50 * 24
    input_reg  = torch.zeros(2, 50, 12)  # B * 50 * 24
    target_cls = torch.zeros(2, 50, 1, dtype=torch.int64)
    target_reg = torch.zeros(2, 50, 1)

    input_cls, target_cls = input_cls.view(-1, 12), target_cls.view(-1)
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')

    a = torch.zeros(2, 24, 10, 10)
    b = torch.zeros(2, 10).long()
    c = torch.ones(2, 10).long()
    d = torch.zeros(2, 10, 1).long()
    e = torch.zeros(2, 10, 1)
    print(compute_heading_loss(a, b, c, d, e))

