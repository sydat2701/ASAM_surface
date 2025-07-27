NUM_EPOCH = 250
thresh_range = [0.3, 0.8]
is_channel_att = True
CA_depth=6

n_res = 1
n_vertices = [153] #num vertices for each patch, corresponding to each resolution
n_patches = [320] #num patches for each resolution
output_units = 1

is_cls_token =True
is_mam = True
is_att = True

input_dims = 3
out_emb_dims = [384] #for each resol
n_states = [16]
n_heads = [8]
MA_depths = [2]

MIN_HOP = [1]
MAX_HOP = [5]

SURV_HOP_THRES_PROB = [0.6]

mam_dep = 1
att_dep = 5