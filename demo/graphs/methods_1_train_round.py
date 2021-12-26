from matplotlib import pyplot as plt
import statistics

if __name__ == "__main__":
    exp_naive = [51.299, 50.661, 52.839, 51.409, 49.693]
    exp_naive_finetune = [51.937, 53.793, 50.28, 51.17, 55.706]
    exp_none = [56.595, 60.151, 58.469, 58.024, 58.957]
    exp_none_finetune = [56.822, 56.691, 59.029, 57.57, 58.111]
    exp_dilation = [60.714, 57.758, 59.404, 61.211, 59.842]
    exp_dilation_finetune = [57.98, 58.171, 59.039, 57.614, 58.343]
    exp_slic_thr1 = [57.957, 58.213, 57.457, 59.423, 59.146]
    exp_slic_thr1_finetune = [55.903, 56.331, 57.004, 55.943, 56.186]
    exp_slic_thr2 = [59.507, 59.289, 59.232, 60.438, 57.425]
    exp_slic_thr2_finetune = [57.777, 58.113, 57.98, 56.54, 55.055]
    exp_grabcut = [57.702, 59.697, 59.018, 57.74, 59.22]
    exp_grabcut_finetune = [54.744, 55.833, 58.285, 55.463, 59.132]

    print(statistics.mean(exp_dilation))
    print(statistics.stdev(exp_dilation))

    names = ["naive", "naive_finetune",
             "none", "none_finetune",
             "dilation", "dilation_finetune",
             "slic_thr1", "slic_thr1_finetune",
             "slic_thr2", "slic_thr2_finetune",
             "grabcut", "grabcut_finetune"]

    means = [statistics.mean(exp_naive), statistics.mean(exp_naive_finetune),
             statistics.mean(exp_none), statistics.mean(exp_none_finetune),
             statistics.mean(exp_dilation), statistics.mean(exp_dilation_finetune),
             statistics.mean(exp_slic_thr1), statistics.mean(exp_slic_thr1_finetune),
             statistics.mean(exp_slic_thr2), statistics.mean(exp_slic_thr2_finetune),
             statistics.mean(exp_grabcut), statistics.mean(exp_grabcut_finetune)]

    stdevs = [statistics.stdev(exp_naive), statistics.stdev(exp_naive_finetune),
             statistics.stdev(exp_none), statistics.stdev(exp_none_finetune),
             statistics.stdev(exp_dilation), statistics.stdev(exp_dilation_finetune),
             statistics.stdev(exp_slic_thr1), statistics.stdev(exp_slic_thr1_finetune),
             statistics.stdev(exp_slic_thr2), statistics.stdev(exp_slic_thr2_finetune),
             statistics.stdev(exp_grabcut), statistics.stdev(exp_grabcut_finetune)]

    # names.insert(0, "baseline w/o crop")
    # means.insert(0, 5.962)
    # stdevs.insert(0, 0)
    #
    # names.insert(1, "baseline")
    # means.insert(1, 39.164)
    # stdevs.insert(1, 0)

    plt.errorbar(names, means, stdevs, linestyle='None', marker='d')
    plt.show()
