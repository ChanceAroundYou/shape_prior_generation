def get_kl_rate(epoch, n=500, m=1000):
    if epoch < m:
        return 0
    else:
        return (epoch % n) / n