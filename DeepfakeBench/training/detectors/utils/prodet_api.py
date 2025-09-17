import torch


def mixup_in_latent_space(domain2, domain1):
    # data shape: [batchsize, num_domains, 3, 256, 256]
    bs = domain1.shape[0]

    # Initialize an empty tensor for mixed data
    mixed_data = torch.zeros_like(domain1)
    ratios = torch.zeros(bs)
    # For each sample in the batch
    for i in range(bs):
        # Step 1: Generate a shuffled index list for the domains

        # Step 2: Choose random alpha between 0.5 and 2, then sample lambda from beta distribution
        alpha = torch.rand(1) * 1.5 + 0.5  # random alpha between 0.5 and 2
        lambda_ = torch.distributions.beta.Beta(alpha, alpha).sample().cuda()

        # Step 3: Perform mixup using the shuffled indices
        mixed_data[i] = lambda_ * domain1[i] + (1 - lambda_) * domain2[i]
        ratios[i] = lambda_

    return mixed_data, ratios


def feature_operation(features):
    single_len = len(features) // 3
    real, blend, fake = features[:single_len], features[single_len:single_len * 2], features[single_len * 2:]
    # in-domain required? pass temp

    # cross domain. TODO: introducing which type of cross besides Mixup？
    # here, ratio used to denote the ratio of the former one, while 1 real should be 0 blend, 1 blend should be 0 df, therefore we re-write the order.
    #  1. real+blend
    real_blends, rb_ratios = mixup_in_latent_space(real, blend)

    # 2. blend+fake
    blend_fakes, bf_ratios = mixup_in_latent_space(blend, fake)
    return real_blends, blend_fakes, rb_ratios, bf_ratios

def feature_operation_v2(features):
    single_len = len(features) // 4
    real, blend, bi, fake = \
        features[:single_len], features[single_len:single_len * 2], features[
                                                                    single_len * 2:single_len * 3], features[
                                                                                                    single_len * 3:single_len * 4]
    # in-domain required? pass temp

    # cross domain. TODO: introducing which type of cross besides Mixup？
    # here, ratio used to denote the ratio of the former one, while 1 real should be 0 blend, 1 blend should be 0 df, therefore we re-write the order.
    #
    real_blends, rb_ratios = mixup_in_latent_space(real, blend)
    blend_bis, bb_ratios = mixup_in_latent_space(blend, bi)
    bi_fakes, bf_ratios = mixup_in_latent_space(bi, fake)
    return real_blends, blend_bis, bi_fakes, rb_ratios, bb_ratios, bf_ratios