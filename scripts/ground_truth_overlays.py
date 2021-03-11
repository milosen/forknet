from forknet.utils.helper import overlay, load_volume, show


if __name__ == '__main__':
    gtGM = load_volume("data/target_GM.pt")
    gtWM = load_volume("data/target_WM.pt")
    t1w = load_volume("data/t1w_test.pt")

    axial_slice = 35

    t1w_slice = t1w[axial_slice].numpy()
    overlay(t1w_slice, gtWM[:, axial_slice].numpy())
    overlay(t1w_slice, gtGM[:, axial_slice].numpy())
    show(t1w_slice)
