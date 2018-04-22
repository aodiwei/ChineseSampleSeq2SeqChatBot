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
    batch_size = 16
    xvocab_size = len(metadata['idx2w'])
    yvocab_size = xvocab_size
    emb_dim = 128
    
    print('xseq_len : {}'.format(xseq_len))
    print('yseq_len : {}'.format(yseq_len))
    print('batch_size : {}'.format(batch_size))
    print('xvocab_size : {}'.format(xvocab_size))
    print('yvocab_size : {}'.format(yvocab_size))
    print('emb_dim : {}'.format(emb_dim))
    print('train : {}'.format(len(trainX)))
    print('test : {}'.format(len(testX)))


    model_seq = Seq2SeqModel(
        xseq_len=xseq_len,
        yseq_len=yseq_len,
        x_vocab_size=xvocab_size,
        y_vocab_size=yvocab_size,
        emb_dim=emb_dim,
        num_layers=2,
        ckpt_path='./model_ckp',
        metadata=metadata,
        batch_size=batch_size,
        mtype='attention',
        # hook=data_process.upload_to_google_drive
    )

    model_seq.train((idx_q, idx_a), (testX, testY))


if __name__ == '__main__':
    run()
