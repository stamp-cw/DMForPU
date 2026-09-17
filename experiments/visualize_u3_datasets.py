"""Create a compact visual audit of the generated GFS/RME datasets."""
from pathlib import Path
import h5py
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    sizes=(128,64,32); families=("GFS","RME"); snrs=(0,5,30)
    fig,axes=plt.subplots(len(families)*len(sizes),len(snrs)+1,figsize=(10,13),constrained_layout=True)
    for fr,family in enumerate(families):
        for sr,size in enumerate(sizes):
            row=fr*len(sizes)+sr; folder=ROOT/"data"/f"{family}{size}"
            with h5py.File(folder/"test_30dB.h5","r") as f: gt=f["phi"][0]
            axes[row,0].imshow(gt,cmap="viridis"); axes[row,0].set_title(f"{family}{size} ground truth")
            for col,snr in enumerate(snrs,1):
                with h5py.File(folder/f"test_{snr}dB.h5","r") as f: wrapped=f["psi"][0]
                axes[row,col].imshow(wrapped,cmap="twilight",vmin=-3.1416,vmax=3.1416)
                axes[row,col].set_title(f"{snr} dB wrapped")
            for ax in axes[row]: ax.axis("off")
    fig.savefig(ROOT/"data"/"DATASETS_V2_preview.png",dpi=180); plt.close(fig)


if __name__=="__main__": main()
