import torch
import torch.nn.functional as F


def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)

    loss = -torch.log(10e-8 + torch.sigmoid(pos_score - neg_score))
    return torch.sum(loss)

def pairwise_marginal_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    zero_tensor = torch.zeros_like(pos_score)
    loss =  torch.mean(torch.max(zero_tensor, 1-pos_score+neg_score))
    return loss

def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.sum(torch.pow(emb, 2))
    return emb_loss * reg

def calcRegLoss(model):
	ret = 0
	for W in model.parameters():
		ret += W.norm(2).square()
	return ret

def batch_softmax_loss(user_emb, item_emb, temperature):
    user_emb, item_emb = F.normalize(user_emb, dim=1), F.normalize(item_emb, dim=1)
    pos_score = (user_emb * item_emb).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(user_emb, item_emb.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    loss = -torch.log(pos_score / ttl_score)
    return torch.mean(loss)


def InfoNCE(view1, view2, temperature):
    view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / ttl_score)
    return torch.sum(cl_loss)


def kl_divergence(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1) - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(kl)


def js_divergence(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    q = F.softmax(q_logit, dim=-1)
    kl_p = torch.sum(p * (F.log_softmax(p_logit, dim=-1) - F.log_softmax(q_logit, dim=-1)), 1)
    kl_q = torch.sum(q * (F.log_softmax(q_logit, dim=-1) - F.log_softmax(p_logit, dim=-1)), 1)
    return torch.mean(kl_p + kl_q)


class BC_LOSS():
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau1 = args.tau1
        self.tau2 = args.tau2
        self.w_lambda = args.w_lambda
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1
        self.n_users_pop=data.n_user_pop
        self.n_items_pop=data.n_item_pop
        self.embed_user_pop = nn.Embedding(self.n_users_pop, self.emb_dim)
        self.embed_item_pop = nn.Embedding(self.n_items_pop, self.emb_dim)
        nn.init.xavier_normal_(self.embed_user_pop.weight)
        nn.init.xavier_normal_(self.embed_item_pop.weight)
    
    def forward(self, users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop):

        # popularity branch
        users_pop_emb = self.embed_user_pop(users_pop)
        pos_pop_emb = self.embed_item_pop(pos_items_pop)
        neg_pop_emb = self.embed_item_pop(neg_items_pop)

        pos_ratings_margin = torch.sum(users_pop_emb * pos_pop_emb, dim = -1)

        users_pop_emb = F.normalize(users_pop_emb, dim = -1)
        pos_pop_emb = F.normalize(pos_pop_emb, dim = -1)
        neg_pop_emb = F.normalize(neg_pop_emb, dim = -1)

        pos_ratings = torch.sum(users_pop_emb * pos_pop_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_pop_emb, 1), 
                                       neg_pop_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)

        numerator = torch.exp(pos_ratings / self.tau2)
        denominator = torch.sum(torch.exp(ratings / self.tau2), dim = 1)
        loss2 = self.w_lambda * torch.mean(torch.negative(torch.log(numerator/denominator)))

        # main branch
        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb = F.normalize(users_emb, dim = -1)
        pos_emb = F.normalize(pos_emb, dim = -1)
        neg_emb = F.normalize(neg_emb, dim = -1)
        
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        pos_ratings = torch.cos(torch.arccos(torch.clamp(pos_ratings,-1+1e-7,1-1e-7))+(1-torch.sigmoid(pos_ratings_margin)))
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), 
                                       neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)

        numerator = torch.exp(pos_ratings / self.tau1)
        denominator = torch.sum(torch.exp(ratings / self.tau1), dim = 1)
        
        loss1 = (1-self.w_lambda) * torch.mean(torch.negative(torch.log(numerator/denominator)))

        # reg loss
        regularizer1 = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + \
                        0.5 * torch.norm(negEmb0) ** 2
        regularizer1 = regularizer1/self.batch_size

        regularizer2= 0.5 * torch.norm(users_pop_emb) ** 2 + 0.5 * torch.norm(pos_pop_emb) ** 2 + \
                        0.5 * torch.norm(neg_pop_emb) ** 2
        regularizer2  = regularizer2/self.batch_size
        reg_loss = self.decay * (regularizer1+regularizer2)

        reg_loss_freeze=self.decay * (regularizer2)
        reg_loss_norm=self.decay * (regularizer1)

        return loss1, loss2, reg_loss, reg_loss_freeze, reg_loss_norm

    def freeze_pop(self):
        self.embed_user_pop.requires_grad_(False)
        self.embed_item_pop.requires_grad_(False)