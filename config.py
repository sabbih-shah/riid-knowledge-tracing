
class CONFIG:
    warmpup_steps = 2500
    heads = 4
    num_layers = 4
    workers = 12
    train_path = "riiid-test-answer-prediction/train.csv"
    pretrained_emb_path = "riid_256_embedding_400.npz"
    train_questions_path = ""
    batch_size = 128
    seq_len = 100
    device = 'cuda'
    emb_dim = 256
    total_questions = 13523
    total_categories = 10000