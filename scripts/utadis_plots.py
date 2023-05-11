import dotenv
import pathlib
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from preference_learning import UtaWrapper, Hook, append_output


def main():
    criteria_nr = 6
    dotenv.load_dotenv()
    project_path = pathlib.Path(os.getenv("PROJECT_PATH"))
    model_path = project_path / "models" / "ann_utadis.pt"
    uta_wrapper = UtaWrapper()
    uta_wrapper.load_model(model_path)
    model = uta_wrapper.model_

    def get_simple_input(val):
        return torch.FloatTensor([[val] * criteria_nr]).view(1, 1, -1).cpu()

    hook = Hook(model.method.criterion_layer_combine, append_output)
    xs = []
    with torch.no_grad():
        for i in range(201):
            val = i / 200.0
            x = get_simple_input(val)
            xs.append(val)
            model(x)

    outs = np.array(torch.stack(hook.stats)[:, 0].detach().cpu())
    outs = outs * model.method.sum_layer.weight.detach().numpy()[0]
    outs = outs[::3] - outs[::3][0]
    outs = outs / outs[-1].sum()

    criteria = ["buying", "maint", "doors", "persons", "lug boot", "safety"]
    for i, criterion in enumerate(criteria):
        plt.plot(xs, outs[:, i], color="black")
        plt.ylabel("$u_{" + criterion + "}$", fontsize=14)
        plt.xlabel("$g_{" + criterion + "}$", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.title("Marginal value function for criterion '{0}'".format(criterion))

        plt.subplots_adjust(top=0.9)

        # Save the plot
        plots_path = pathlib.Path(os.getenv("PROJECT_PATH")) / "plots"
        plt.savefig(plots_path / f"g{i + 1}.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
