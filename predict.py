import argparse
import torch
import yaml
import dgl
import numpy as np
from utils.graph_path import path_complex_mol
from model.ka_gat import KA_GAT
from model.mlp_gat import MLP_GAT
from model.kan_gat import KAN_GAT
from model.po_gat import PO_GAT


# -----------------------------
# Label Names
# -----------------------------
def get_label_names(task):
    if task == "tox21":
        return ['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD',
                'NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']
    if task == "clintox":
        return ['FDA_APPROVED','CT_TOX']
    if task == "sider":
        return ['Hepatobiliary disorders','Metabolism and nutrition disorders',
                'Product issues','Eye disorders','Investigations',
                'Musculoskeletal and connective tissue disorders',
                'Gastrointestinal disorders','Social circumstances',
                'Immune system disorders','Reproductive system and breast disorders',
                'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
                'General disorders and administration site conditions','Endocrine disorders',
                'Surgical and medical procedures','Vascular disorders',
                'Blood and lymphatic system disorders','Skin disorders',
                'Congenital disorders','Infections','Respiratory disorders',
                'Psychiatric disorders','Renal disorders',
                'Pregnancy conditions','Ear disorders','Cardiac disorders',
                'Nervous system disorders','Injury/poisoning']
    if task == "muv":
        return ['MUV-466','MUV-548','MUV-600','MUV-644','MUV-652','MUV-689','MUV-692',
                'MUV-712','MUV-713','MUV-733','MUV-737','MUV-810','MUV-832',
                'MUV-846','MUV-852','MUV-858','MUV-859']
    if task in ["bbbp"]:
        return ["bbbp"]
    return []


# -----------------------------
# Target Dimension
# -----------------------------
def get_target_dim(task):
    return {
        'tox21':12,'muv':17,'sider':27,'clintox':2,
        'bace':1,'bbbp':1,'hiv':1
    }[task]



# -----------------------------
# Model Selection Dictionary (same style as training script)
# -----------------------------
TASK_MODEL = {
    "tox21":   "kagat",
    "sider":   "kagat",
    "clintox": "kagat",
    "bbbp":    "kagat",
}


# ===========================================================
#                    MAIN INFERENCE
# ===========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--config", default="./config/gat_path.yaml")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode (ignore GPU)")
    args = parser.parse_args()

    task = args.task.lower()

    # ---------------- Load config ----------------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    encoder_atom = cfg["encoder_atom"]
    encoder_bond = cfg["encoder_bond"]
    head = cfg["head"]
    num_layers = cfg["num_layers"]
    pooling = cfg["pooling"]
    grid = cfg["grid"]

    # ---------------- Setup device ----------------
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---------------- Auto model path ----------------
    if args.model_path is None:
        args.model_path = f"./models/{task}.pth"
    print("Loading:", args.model_path)

    # ---------------- Graph build ----------------
    g = path_complex_mol(args.smiles, encoder_atom, encoder_bond)
    if g is False:
        print("Could not process SMILES.")
        return

    g = g.to(device)
    node_f = g.ndata["feat"].to(device)
    edge_f = g.edata["feat"].to(device)

    encode_dim = [92, 21]  # SAME as training

    # ---------------- Model selection ----------------
    model_choice = TASK_MODEL[task]
    # print(f"Task {task} → using model {model_choice}")

    out_dim = get_target_dim(task)

    if model_choice == "kagat":
        model = KA_GAT(encode_dim[0], encode_dim[1], 64, 32, out_dim,
                       grid, head, num_layers, pooling)
    elif model_choice == "kangat":
        model = KAN_GAT(encode_dim[0], encode_dim[1], 64, 32, out_dim,
                        grid, head, num_layers, pooling)
    elif model_choice == "mlpgat":
        model = MLP_GAT(encode_dim[0], encode_dim[1], 64, 32, out_dim,
                        grid, head, num_layers, pooling)
    elif model_choice == "pogat":
        model = PO_GAT(encode_dim[0], encode_dim[1], 64, 32, out_dim,
                       grid, head, num_layers, pooling)
    else:
        raise ValueError("Unknown model type")

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # ---------------- Predict ----------------
    with torch.no_grad():
        out = model(g, node_f, edge_f).cpu().numpy().flatten()

    names = get_label_names(task)

    print("\n=========== Prediction ===========")
    for i, name in enumerate(names):
        print(f"{name:40s}: {out[i]:.4f}")


if __name__ == "__main__":
    main()
