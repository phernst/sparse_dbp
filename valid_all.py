from valid_fdkconvnet import main as v_fdk
from valid_hilbert_inverse_3chan import main as v_hi3
from valid_hilbert_inverse_sparse_3chan import main as v_his3
from valid_hilbert_inverse_sparse import main as v_his
from valid_hilbert_inverse import main as v_hi


def main():
    v_fdk()
    v_hi3()
    v_his3()
    v_his()
    v_hi()


if __name__ == "__main__":
    main()
