from slowfast.utils.meters import ClevrerTrainMeter, ClevrerValMeter
from slowfast.config.defaults import get_cfg

def test_train_iter1():
    cfg = get_cfg()
    cfg.LOG_PERIOD = 1
    train_meter = ClevrerTrainMeter(5, cfg)
    train_meter.update_stats(top1_err=40.0, top5_err=20.0, mc_opt_err=55.0, mc_q_err=78.0, 
        loss_des=6.3, loss_mc=0.2, lr=0.1, mb_size_des=10, mb_size_mc=10)
    stats = train_meter.log_iter_stats(cur_epoch=1, cur_iter=1)
    assert stats['top1_err'] == 40.0
    assert stats['top5_err'] == 20.0
    assert stats['mc_opt_err'] == 55.0
    assert stats['mc_q_err'] == 78.0
    assert stats['loss_des'] == 6.3
    assert stats['loss_mc'] == 0.2

def test_train_iter2():
    cfg = get_cfg()
    cfg.LOG_PERIOD = 1
    train_meter = ClevrerTrainMeter(5, cfg)
    train_meter.update_stats(top1_err=40.0, top5_err=20.0, mc_opt_err=55.0, mc_q_err=78.0, 
        loss_des=6.3, loss_mc=0.2, lr=0.1, mb_size_des=10, mb_size_mc=10)
    train_meter.update_stats(top1_err=41.0, top5_err=21.0, mc_opt_err=56.0, mc_q_err=86.0, 
        loss_des=6.7, loss_mc=1.2, lr=0.1, mb_size_des=10, mb_size_mc=10)
    stats = train_meter.log_iter_stats(cur_epoch=1, cur_iter=1)
    assert stats['top1_err'] == 41.0
    assert stats['top5_err'] == 21.0
    assert stats['mc_opt_err'] == 56.0
    assert stats['mc_q_err'] == 86.0
    assert stats['loss_des'] == 6.7
    assert stats['loss_mc'] == 1.2

def test_train_iter3():
    cfg = get_cfg()
    cfg.LOG_PERIOD = 2
    train_meter = ClevrerTrainMeter(5, cfg)
    train_meter.update_stats(top1_err=40.0, top5_err=20.0, mc_opt_err=55.0, mc_q_err=78.0, 
        loss_des=6.3, loss_mc=0.2, lr=0.1, mb_size_des=10, mb_size_mc=10)
    train_meter.update_stats(top1_err=41.0, top5_err=21.0, mc_opt_err=56.0, mc_q_err=86.0, 
        loss_des=6.7, loss_mc=1.2, lr=0.1, mb_size_des=10, mb_size_mc=10)
    stats = train_meter.log_iter_stats(cur_epoch=1, cur_iter=2)
    assert stats is None

def test_train_iter4():
    cfg = get_cfg()
    cfg.LOG_PERIOD = 2
    train_meter = ClevrerTrainMeter(5, cfg)
    train_meter.update_stats(top1_err=40.0, top5_err=20.0, mc_opt_err=55.0, mc_q_err=78.0, 
        loss_des=6.3, loss_mc=0.2, lr=0.1, mb_size_des=10, mb_size_mc=10)
    train_meter.update_stats(top1_err=10.0, top5_err=10.0, mc_opt_err=10.0, mc_q_err=10.0, 
        loss_des=10.0, loss_mc=10.0, lr=0.1, mb_size_des=10, mb_size_mc=10)
    stats = train_meter.log_iter_stats(cur_epoch=1, cur_iter=2)
    assert stats is None
    train_meter.update_stats(top1_err=20.0, top5_err=20.0, mc_opt_err=20.0, mc_q_err=20.0, 
        loss_des=20.0, loss_mc=20.0, lr=0.1, mb_size_des=10, mb_size_mc=10)
    stats = train_meter.log_iter_stats(cur_epoch=0, cur_iter=3)
    assert stats['top1_err'] == 15.0
    assert stats['top5_err'] == 15.0
    assert stats['mc_opt_err'] == 15.0
    assert stats['mc_q_err'] == 15.0
    assert stats['loss_des'] == 15.0
    assert stats['loss_mc'] == 15.0

def test_train_iter5():
    cfg = get_cfg()
    cfg.LOG_PERIOD = 3
    train_meter = ClevrerTrainMeter(5, cfg)
    train_meter.update_stats(top1_err=30.0, top5_err=30.0, mc_opt_err=30.0, mc_q_err=30.0, 
        loss_des=30.0, loss_mc=30.0, lr=0.1, mb_size_des=10, mb_size_mc=10)
    train_meter.update_stats(top1_err=10.0, top5_err=10.0, mc_opt_err=10.0, mc_q_err=10.0, 
        loss_des=10.0, loss_mc=10.0, lr=0.1, mb_size_des=10, mb_size_mc=10)
    stats = train_meter.log_iter_stats(cur_epoch=1, cur_iter=1)
    assert stats is None
    train_meter.update_stats(top1_err=20.0, top5_err=20.0, mc_opt_err=20.0, mc_q_err=20.0, 
        loss_des=20.0, loss_mc=20.0, lr=0.1, mb_size_des=10, mb_size_mc=10)
    stats = train_meter.log_iter_stats(cur_epoch=0, cur_iter=5)
    assert stats['top1_err'] == 20.0, print(stats['top1_err'])
    assert stats['top5_err'] == 20.0
    assert stats['mc_opt_err'] == 20.0
    assert stats['mc_q_err'] == 20.0
    assert stats['loss_des'] == 20.0
    assert stats['loss_mc'] == 20.0

def test_train_epoch_only_des():
    cfg = get_cfg()
    cfg.LOG_PERIOD = 3
    train_meter = ClevrerTrainMeter(5, cfg)
    train_meter.update_stats(top1_err=30.0, top5_err=30.0, mc_opt_err=30.0, mc_q_err=30.0, 
        loss_des=30.0, loss_mc=30.0, lr=0.1, mb_size_des=10, mb_size_mc=0)
    train_meter.update_stats(top1_err=10.0, top5_err=10.0, mc_opt_err=10.0, mc_q_err=10.0, 
        loss_des=10.0, loss_mc=10.0, lr=0.1, mb_size_des=10, mb_size_mc=0)
    train_meter.update_stats(top1_err=20.0, top5_err=20.0, mc_opt_err=20.0, mc_q_err=20.0, 
        loss_des=20.0, loss_mc=20.0, lr=0.1, mb_size_des=10, mb_size_mc=0)
    stats = train_meter.log_epoch_stats(cur_epoch=1)
    assert stats['top1_err'] == 20.0, print(stats['top1_err'])
    assert stats['top5_err'] == 20.0
    assert not 'mc_opt_err' in stats
    assert not 'mc_q_err' in stats
    assert stats['loss_des'] == 20.0
    assert not 'loss_mc' in stats

def test_train_epoch():
    cfg = get_cfg()
    cfg.LOG_PERIOD = 3
    train_meter = ClevrerTrainMeter(5, cfg)
    train_meter.update_stats(top1_err=30.0, top5_err=30.0, mc_opt_err=60.0, mc_q_err=60.0, 
        loss_des=30.0, loss_mc=60.0, lr=0.1, mb_size_des=10, mb_size_mc=5)
    train_meter.update_stats(top1_err=10.0, top5_err=10.0, mc_opt_err=10.0, mc_q_err=10.0, 
        loss_des=10.0, loss_mc=10.0, lr=0.1, mb_size_des=10, mb_size_mc=5)
    train_meter.update_stats(top1_err=20.0, top5_err=20.0, mc_opt_err=20.0, mc_q_err=20.0, 
        loss_des=20.0, loss_mc=20.0, lr=0.1, mb_size_des=10, mb_size_mc=5)
    stats = train_meter.log_epoch_stats(cur_epoch=1)
    assert stats['top1_err'] == 20.0, print(stats['top1_err'])
    assert stats['top5_err'] == 20.0
    assert stats['mc_opt_err'] == 30.0
    assert stats['mc_q_err'] == 30.0
    assert stats['loss_des'] == 20.0
    assert stats['loss_mc'] == 30.0

    train_meter.reset()
    train_meter.update_stats(top1_err=30.0, top5_err=30.0, mc_opt_err=30.0, mc_q_err=30.0, 
        loss_des=30.0, loss_mc=30.0, lr=0.1, mb_size_des=10, mb_size_mc=5)
    train_meter.update_stats(top1_err=10.0, top5_err=10.0, mc_opt_err=10.0, mc_q_err=10.0, 
        loss_des=10.0, loss_mc=10.0, lr=0.1, mb_size_des=10, mb_size_mc=5)
    train_meter.update_stats(top1_err=20.0, top5_err=20.0, mc_opt_err=20.0, mc_q_err=20.0, 
        loss_des=20.0, loss_mc=20.0, lr=0.1, mb_size_des=10, mb_size_mc=5)
    stats = train_meter.log_epoch_stats(cur_epoch=2)
    assert stats['top1_err'] == 20.0, print(stats['top1_err'])
    assert stats['top5_err'] == 20.0
    assert stats['mc_opt_err'] == 20.0
    assert stats['mc_q_err'] == 20.0
    assert stats['loss_des'] == 20.0
    assert stats['loss_mc'] == 20.0

def test_val_epoch():
    cfg = get_cfg()
    cfg.LOG_PERIOD = 4
    val_meter = ClevrerValMeter(5, cfg)
    val_meter.update_stats(top1_err=30.0, top5_err=30.0, mc_opt_err=60.0, mc_q_err=60.0, 
        loss_des=30.0, loss_mc=60.0, mb_size_des=10, mb_size_mc=5)
    val_meter.update_stats(top1_err=10.0, top5_err=10.0, mc_opt_err=10.0, mc_q_err=10.0, 
        loss_des=10.0, loss_mc=10.0, mb_size_des=10, mb_size_mc=5)
    val_meter.update_stats(top1_err=20.0, top5_err=20.0, mc_opt_err=20.0, mc_q_err=20.0, 
        loss_des=20.0, loss_mc=20.0, mb_size_des=10, mb_size_mc=5)
    stats = val_meter.log_epoch_stats(cur_epoch=1)
    assert stats['top1_err'] == 20.0, print(stats['top1_err'])
    assert stats['top5_err'] == 20.0
    assert stats['mc_opt_err'] == 30.0
    assert stats['mc_q_err'] == 30.0
    assert stats['loss_des'] == 20.0
    assert stats['loss_mc'] == 30.0

    val_meter.reset()
    val_meter.update_stats(top1_err=60.0, top5_err=30.0, mc_opt_err=30.0, mc_q_err=30.0, 
        loss_des=30.0, loss_mc=30.0, mb_size_des=10, mb_size_mc=5)
    val_meter.update_stats(top1_err=10.0, top5_err=10.0, mc_opt_err=10.0, mc_q_err=10.0, 
        loss_des=10.0, loss_mc=10.0, mb_size_des=10, mb_size_mc=5)
    val_meter.update_stats(top1_err=20.0, top5_err=20.0, mc_opt_err=20.0, mc_q_err=20.0, 
        loss_des=20.0, loss_mc=20.0, mb_size_des=10, mb_size_mc=5)
    stats = val_meter.log_epoch_stats(cur_epoch=2)
    assert stats['top1_err'] == 30.0, print(stats['top1_err'])
    assert stats['top5_err'] == 20.0
    assert stats['mc_opt_err'] == 20.0
    assert stats['mc_q_err'] == 20.0
    assert stats['loss_des'] == 20.0
    assert stats['loss_mc'] == 20.0

if __name__ == "__main__":
    test_train_iter1()
    print("test_train_iter1 passed")
    test_train_iter2()
    print("test_train_iter2 passed")
    test_train_iter3()
    print("test_train_iter3 passed")
    test_train_iter4()
    print("test_train_iter4 passed")
    test_train_iter5()
    print("test_train_iter5 passed")
    test_train_epoch_only_des()
    print("test_train_epoch_only_des passed")
    test_train_epoch()
    print("test_train_epoch passed")
    test_val_epoch()
    print("test_val_epoch passed")
    