class sequence:
    def shutdown(self):
        pass
    def __init__(self, seq_num, *args, **kwargs):
        self.seq_num = seq_num
        self.sequence = np.zeros((seq_num,144,256,3))
    def run(self, img_arr):
        for i in range(seq_num-1):
            sequence[i] = sequence[i+1]
        sequence[seq_num-1] = img_arr
        return sequence