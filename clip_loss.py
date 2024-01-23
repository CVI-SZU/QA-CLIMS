import torch


class InfoNCELossFG(torch.nn.Module):
    def __init__(self, temperature=1.0, ):
        super(InfoNCELossFG, self).__init__()
        self.temperature = temperature
        print(f'Use InfoNCELossFG '
              f'temperature: {temperature} ')

    def forward(self, fg_img_feature, fg_text_feature, bg_text_feature):
        positive_sims = torch.tensor(0., requires_grad=True, device=fg_img_feature.device)
        negative_sims = torch.tensor(0., requires_grad=True, device=fg_img_feature.device)

        fg_img_feature = fg_img_feature / fg_img_feature.norm(dim=-1, keepdim=True)

        fg_img_fg_text_logits = fg_img_feature @ fg_text_feature.t()  # [1, 1]
        fg_img_bg_text_logits = fg_img_feature @ bg_text_feature.t()  # [1, L]

        positive_sims = positive_sims + torch.exp(fg_img_fg_text_logits / self.temperature).sum()
        negative_sims = negative_sims + \
                        torch.exp(fg_img_fg_text_logits / self.temperature).sum() + \
                        torch.exp(fg_img_bg_text_logits / self.temperature).sum()

        loss = -torch.log(positive_sims / negative_sims)

        return loss


class InfoNCELossBG(torch.nn.Module):
    def __init__(self, temperature=1.0, ):
        super(InfoNCELossBG, self).__init__()
        self.temperature = temperature
        print(f'Use InfoNCELossBG '
              f'temperature: {temperature} ')

    def forward(self, bg_img_feature, fg_text_feature, bg_text_feature):
        positive_sims = torch.tensor(0., requires_grad=True, device=bg_img_feature.device)
        negative_sims = torch.tensor(0., requires_grad=True, device=bg_img_feature.device)

        bg_img_feature = bg_img_feature / bg_img_feature.norm(dim=-1, keepdim=True)

        bg_img_bg_text_logits = bg_img_feature @ bg_text_feature.t()  # [1, L]
        bg_img_fg_text_logits = bg_img_feature @ fg_text_feature.t()  # [1, 1]

        positive_sims = positive_sims + torch.exp(bg_img_bg_text_logits / self.temperature).mean()
        negative_sims = negative_sims + \
                        torch.exp(bg_img_bg_text_logits / self.temperature).mean() + \
                        torch.exp(bg_img_fg_text_logits / self.temperature).sum()

        loss = -torch.log(positive_sims / negative_sims)

        return loss
