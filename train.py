import tensorflow as tf

from model import Seq2SeqModel
import data_process


def run():
    """
    running
    :return:
    """
    idx_q, idx_a = data_process.load_qa_data()
    metadata = data_process.load_metadata()
    (trainX, trainY), (testX, testY) = data_process.split_dataset(idx_q, idx_a)

    xseq_len = trainX.shape[-1]
    yseq_len = trainY.shape[-1]
    batch_size = 32
    xvocab_size = len(metadata['idx2w'])
    yvocab_size = xvocab_size
    emb_dim = 512

    model = Seq2SeqModel(
        xseq_len=xseq_len,
        yseq_len=yseq_len,
        x_vocab_size=xvocab_size,
        y_vocab_size=yvocab_size,
        emb_dim=emb_dim,
        num_layers=2,
        ckpt_path='./model_ckp',
        metadata=metadata,
        batch_size=batch_size,
        hook=data_process.upload_to_google_drive
    )

    model.train((trainX, trainY), (testX, testY))


if __name__ == '__main__':
    run()
