from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

# Dictionary of PC-lipid conjugates and their SMILES
smiles_dict = {
    "PC-DHA": "OCC(CO)(CCCCNC(=O)C(NC(=O)CCC=CCC=CCC=CCC=CCC=CCC=CCC)COP(=O)(O)(OCC[N+](C)(C)(C)))",
    "DHA": "OCC(CO)(CCCCNC(=O)(CCC=CCC=CCC=CCC=CCC=CCC=CCC))",
    "DCA": "OCC(CO)(CCCCNC(=O)(CCCCCCCCCCCCCCCCCCCCC))",
    "PC-DCA": "OCC(CO)(CCCCNC(=O)C(NC(=O)CCCCCCCCCCCCCCCCCCCCC)COP(=O)(O)(OCC[N+](C)(C)(C)))",
    "DCA_Phos": "OCC(COP([*])([OH])=O)CCCCNC(=O)CCCCCCCCCCCCCCCCCCCCC",
    "Dendv2": "[O-]P(OC(COP(OCCCCCCOP(OC(COP([O-])(OCCCCCCCCCCCCO)=O)COP([O-])(OCCCCCCCCCCCCO)=O)([O-])=O)([O-])=O)COP([O-])(OCCCCCCOP([O-])(OC(COP([O-])(OCCCCCCCCCCCCO)=O)COP([O-])(OCCCCCCCCCCCCO)=O)=O)=O)([*])=O",
    "Dend": "P(=O)(O)(OC(COP(=O)(O)OCCCCCCOP(=O)(O)O(C(COP(=O)(O)OCCCCCCCCCCCCO)COP(=O)(O)OCCCCCCCCCCCCO))COP(=O)(O)OCCCCCCOP(=O)(O)OC(COP(=O)(O)OCCCCCCCCCCCCO)COP(=O)(O)OCCCCCCCCCCCCO)",
    "Chol": "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)OC(=O)NCCCCC(CO)CO)C)C",
    "PC-Chol": "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)OC(=O)NC(COP(=O)([O-])OCC[N+](C)(C)(C))C(=O)NCCCCC(CO)CO)C)C",
    "EPA": "OCC(CO)(CCCCNC(=O)(CCCC=CCC=CCC=CCC=CCC=CCC))",
    "PC-EPA": "OCC(CO)(CCCCNC(=O)C(NC(=O)CCCC=CCC=CCC=CCC=CCC=CCC)COP(=O)([O-])(OCC[N+](C)(C)(C)))",
    "LA": "C[C@H](CCC(=O)NCCCCC(CO)CO)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC[C@H]4[C@@]3(CC[C@H](C4)O)C)C",
    "PC-LA": "C[C@H](CCC(=O)NC(COP(=O)([O-])OCC[N+](C)(C)(C))C(=O)NCCCCC(CO)CO)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC[C@H]4[C@@]3(CC[C@H](C4)O)C)C",
    "RA": "CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C/C(=C/C(=O)NCCCCC(CO)CO)/C)/C",
    "PC-RA": "CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C/C(=C/C(=O)NC(COP(=O)([O-])OCC[N+](C)(C)(C))C(=O)NCCCCC(CO)CO)/C)/C",
    "Oligo1": "O=C(C=CN1[C@@]2([H])[C@@](OC)([H])[C@@](OP(OC[C@@]3([H])O[C@](N(C=N4)C5=C4C(N)=NC=N5)([H])[C@@](F)([H])[C@]3([H])OP(OC[C@@]6([H])O[C@](N(C=N7)C8=C7C(N)=NC=N8)([H])[C@@](OC)([H])[C@]6([H])OP(OC[C@@]9([H])O[C@](N(C(N%10)=O)C=CC%10=O)([H])[C@@](F)([H])[C@]9([H])OP(OC[C@@]%11([H])O[C@](N(C=CC(N)=N%12)C%12=O)([H])[C@@](F)([H])[C@]%11([H])OP(OC[C@@]%13([H])O[C@](N(C=N%14)C%15=C%14C(NC(N)=N%15)=O)([H])[C@@](F)([H])[C@]%13([H])OP(OC[C@@]%16([H])O[C@](N(C(N%17)=O)C=CC%17=O)([H])[C@@](OC)([H])[C@]%16([H])OP(OC[C@@]%18([H])O[C@](N(C=N%19)C%20=C%19C(N)=NC=N%20)([H])[C@@](F)([H])[C@]%18([H])OP(OC[C@@]%21([H])O[C@](N(C(N%22)=O)C=CC%22=O)([H])[C@@](OC)([H])[C@]%21([H])OP(OC[C@@]%23([H])O[C@](N(C(N%24)=O)C=CC%24=O)([H])[C@@](F)([H])[C@]%23([H])OP(OC[C@@]%25([H])O[C@](N(C(N%26)=O)C=CC%26=O)([H])[C@@](OC)([H])[C@]%25([H])O)([O-])=O)([O-])=O)([O-])=O)([O-])=O)([O-])=O)([O-])=O)([O-])=O)([O-])=O)([O-])=S)([O-])=S)([H])[C@](CO)([H])O2)NC1=O",
    "Oligo2": "O=C(C=CN1[C@@]2([H])[C@@](OC)([H])[C@@](OP(OC[C@@]3([H])O[C@](N(C=N4)C5=C4C(N)=NC=N5)([H])[C@@](F)([H])[C@]3([H])OP(OC[C@@]6([H])O[C@](N(C=N7)C8=C7C(N)=NC=N8)([H])[C@@](OC)([H])[C@]6([H])OP(OC[C@@]9([H])O[C@](N(C(N%10)=O)C=CC%10=O)([H])[C@@](F)([H])[C@]9([H])OP(OC[C@@]%11([H])O[C@](N(C=CC(N)=N%12)C%12=O)([H])[C@@](F)([H])[C@]%11([H])OP(OC[C@@]%13([H])O[C@](N(C=N%14)C%15=C%14C(NC(N)=N%15)=O)([H])[C@@](F)([H])[C@]%13([H])OP(OC[C@@]%16([H])O[C@](N(C(N%17)=O)C=CC%17=O)([H])[C@@](OC)([H])[C@]%16([H])OP(OC[C@@]%18([H])O[C@](N(C=N%19)C%20=C%19C(N)=NC=N%20)([H])[C@@](F)([H])[C@]%18([H])OP(OC[C@@]%21([H])O[C@](N(C(N%22)=O)C=CC%22=O)([H])[C@@](OC)([H])[C@]%21([H])OP(OC[C@@]%23([H])O[C@](N(C(N%24)=O)C=CC%24=O)([H])[C@@](F)([H])[C@]%23([H])OP([O-])(OC[C@@]%25([H])O[C@](N(C=N%26)C%27=C%26C(NC(N)=N%27)=O)([H])[C@@](F)([H])[C@]%25([H])O)=O)([O-])=O)([O-])=O)([O-])=O)([O-])=O)([O-])=O)([O-])=O)([O-])=O)([O-])=S)([O-])=S)([H])[C@](CO)([H])O2)NC1=O",
    "OligoLnaG": "O=C(C=CN1[C@@]2([H])[C@@](OC)([H])[C@@](OP(OC[C@@]3([H])O[C@](N(C=N4)C5=C4C(N)=NC=N5)([H])[C@@](F)([H])[C@]3([H])OP(OC[C@@]6([H])O[C@](N(C=N7)C8=C7C(N)=NC=N8)([H])[C@@](OC)([H])[C@]6([H])OP(OC[C@@]9([H])O[C@](N(C(N%10)=O)C=CC%10=O)([H])[C@@](F)([H])[C@]9([H])OP(OC[C@@]%11([H])O[C@](N(C=CC(N)=N%12)C%12=O)([H])[C@@](F)([H])[C@]%11([H])OP(OC[C@@]%13([H])O[C@](N(C=N%14)C%15=C%14C(NC(N)=N%15)=O)([H])[C@@](F)([H])[C@]%13([H])OP(OC[C@@]%16([H])O[C@](N(C(N%17)=O)C=CC%17=O)([H])[C@@](OC)([H])[C@]%16([H])OP(OC[C@@]%18([H])O[C@](N(C=N%19)C%20=C%19C(N)=NC=N%20)([H])[C@@](F)([H])[C@]%18([H])OP(OC[C@@]%21([H])O[C@](N(C(N%22)=O)C=CC%22=O)([H])[C@@](OC)([H])[C@]%21([H])OP(OC[C@@]%23([H])O[C@](N(C(N%24)=O)C=CC%24=O)([H])[C@@](F)([H])[C@]%23([H])OP([O-])(OC[C@@]%25%26O[C@](N(C=N%27)C%28=C%27C(NC(N)=N%28)=O)([H])[C@@](OC%26)([H])[C@]%25([H])O)=O)([O-])=O)([O-])=O)([O-])=O)([O-])=O)([O-])=O)([O-])=O)([O-])=O)([O-])=S)([O-])=S)([H])[C@](CO)([H])O2)NC1=O",
    "Oligo2fGfU": "O=C(C=CN1[C@@]2([H])[C@@](OC)([H])[C@@](OP(OC[C@@]3([H])O[C@](N(C=N4)C5=C4C(N)=NC=N5)([H])[C@@](F)([H])[C@]3([H])OP(OC[C@@]6([H])O[C@](N(C=N7)C8=C7C(N)=NC=N8)([H])[C@@](OC)([H])[C@]6([H])OP(OC[C@@]9([H])O[C@](N(C(N%10)=O)C=CC%10=O)([H])[C@@](F)([H])[C@]9([H])OP(OC[C@@]%11([H])O[C@](N(C=CC(N)=N%12)C%12=O)([H])[C@@](F)([H])[C@]%11([H])OP(OC[C@@]%13([H])O[C@](N(C=N%14)C%15=C%14C(NC(N)=N%15)=O)([H])[C@@](F)([H])[C@]%13([H])OP(OC[C@@]%16([H])O[C@](N(C(N%17)=O)C=CC%17=O)([H])[C@@](OC)([H])[C@]%16([H])OP(OC[C@@]%18([H])O[C@](N(C=N%19)C%20=C%19C(N)=NC=N%20)([H])[C@@](F)([H])[C@]%18([H])OP(OC[C@@]%21([H])O[C@](N(C(N%22)=O)C=CC%22=O)([H])[C@@](OC)([H])[C@]%21([H])OP([O-])(OC[C@@]%23([H])O[C@](N(C=N%24)C%25=C%24C(NC(N)=N%25)=O)([H])[C@@](F)([H])[C@]%23([H])OP(OC[C@@]%26([H])O[C@](N(C(N%27)=O)C=CC%27=O)([H])[C@@](F)([H])[C@]%26([H])O)([O-])=O)=O)([O-])=O)([O-])=O)([O-])=O)([O-])=O)([O-])=O)([O-])=O)([O-])=S)([O-])=S)([H])[C@](CO)([H])O2)NC1=O",
    "AS_highPS": "[H]O[C@@H]1[C@@H](F)[C@H](n2ccc(=O)[nH]c2=O)O[C@@H]1COP(=O)(O)O[C@@H]1[C@@H](OC)[C@H](n2ccc(=O)[nH]c2=O)O[C@@H]1COP(O)(=S)O[C@@H]1[C@@H](OC)[C@H](n2ccc(=O)[nH]c2=O)O[C@@H]1COP(O)(=S)O[C@@H]1[C@@H](OC)[C@H](n2cnc3c(N)ncnc32)O[C@@H]1COP(O)(=S)O[C@@H]1[C@@H](OC)[C@H](n2cnc3c(N)ncnc32)O[C@@H]1COP(O)(=S)O[C@@H]1[C@@H](F)[C@H](n2ccc(=O)[nH]c2=O)O[C@@H]1COP(O)(=S)O[C@@H]1[C@@H](OC)[C@H](n2cnc3c(N)ncnc32)O[C@@H]1COP(O)(=S)O[C@@H]1[C@@H](F)[C@H](n2ccc(=O)[nH]c2=O)O[C@@H]1COP(O)(=S)O[C@@H]1[C@@H](OC)[C@H](n2ccc(N)nc2=O)O[C@@H]1COP(=O)(O)O[C@@H]1[C@@H](F)[C@H](n2cnc3c(N)ncnc32)O[C@@H]1COP(=O)(O)O[C@@H]1[C@@H](OC)[C@H](n2ccc(=O)[nH]c2=O)O[C@@H]1COP(=O)(O)O[C@@H]1[C@@H](F)[C@H](n2ccc(=O)[nH]c2=O)O[C@@H]1COP(=O)(O)O[C@@H]1[C@@H](OC)[C@H](n2ccc(N)nc2=O)O[C@@H]1COP(=O)(O)O[C@@H]1[C@@H](F)[C@H](n2ccc(=O)[nH]c2=O)O[C@@H]1COP(=O)(O)O[C@@H]1[C@@H](OC)[C@H](n2ccc(=O)[nH]c2=O)O[C@@H]1COP(=O)(O)O[C@@H]1[C@@H](F)[C@H](n2cnc3c(=O)[nH]c(N)nc32)O[C@@H]1COP(=O)(O)O[C@@H]1[C@@H](F)[C@H](n2ccc(=O)[nH]c2=O)O[C@@H]1COP(=O)(O)O[C@@H]1[C@@H](F)[C@H](n2cnc3c(N)ncnc32)O[C@@H]1COP(=O)(O)O[C@@H]1[C@@H](OC)[C@H](n2cnc3c(=O)[nH]c(N)nc32)O[C@@H]1COP(=O)(O)O[C@@H]1[C@@H](F)[C@H](n2cnc3c(N)ncnc32)O[C@@H]1COP(O)(=S)O[C@@H]1[C@@H](OC)[C@H](n2ccc(=O)[nH]c2=O)O[C@@H]1COP(O)(O)=S",
    "AS_lowPS": "[H]O[C@@H]1[C@@H](F)[C@H](n2ccc(=O)[nH]c2=O)O[C@@H]1COP(=O)(O)O[C@@H]1[C@@H](OC)[C@H](n2ccc(=O)[nH]c2=O)O[C@@H]1COP(O)(=S)O[C@@H]1[C@@H](OC)[C@H](n2ccc(=O)[nH]c2=O)O[C@@H]1COP(O)(=S)O[C@H]1[C@H](OC)[C@@H](n2cnc3c(N)ncnc32)O[C@H]1COP(=O)(O)O[C@H]1[C@H](OC)[C@@H](n2cnc3c(N)ncnc32)O[C@H]1COP(=O)(O)O[C@@H]1[C@@H](F)[C@H](n2ccc(=O)[nH]c2=O)O[C@@H]1COP(=O)(O)O[C@H]1[C@H](OC)[C@@H](n2cnc3c(N)ncnc32)O[C@H]1COP(=O)(O)O[C@@H]1[C@@H](F)[C@H](n2ccc(=O)[nH]c2=O)O[C@@H]1COP(=O)(O)O[C@@H]1[C@@H](OC)[C@H](n2ccc(N)nc2=O)O[C@@H]1COP(=O)(O)O[C@@H]1[C@@H](F)[C@H](n2cnc3c(N)ncnc32)O[C@@H]1COP(=O)(O)O[C@@H]1[C@@H](OC)[C@H](n2ccc(=O)[nH]c2=O)O[C@@H]1COP(=O)(O)O[C@@H]1[C@@H](F)[C@H](n2ccc(=O)[nH]c2=O)O[C@@H]1COP(=O)(O)O[C@@H]1[C@@H](OC)[C@H](n2ccc(N)nc2=O)O[C@@H]1COP(=O)(O)O[C@@H]1[C@@H](F)[C@H](n2ccc(=O)[nH]c2=O)O[C@@H]1COP(=O)(O)O[C@@H]1[C@@H](OC)[C@H](n2ccc(=O)[nH]c2=O)O[C@@H]1COP(=O)(O)O[C@@H]1[C@@H](F)[C@H](n2cnc3c(=O)[nH]c(N)nc32)O[C@@H]1COP(=O)(O)O[C@@H]1[C@@H](F)[C@H](n2ccc(=O)[nH]c2=O)O[C@@H]1COP(=O)(O)O[C@@H]1[C@@H](F)[C@H](n2cnc3c(N)ncnc32)O[C@@H]1COP(=O)(O)O[C@@H]1[C@@H](OC)[C@H](n2cnc3c(=O)[nH]c(N)nc32)O[C@@H]1COP(=O)(O)O[C@@H]1[C@@H](F)[C@H](n2cnc3c(N)ncnc32)O[C@@H]1COP(O)(=S)O[C@@H]1[C@@H](OC)[C@H](n2ccc(=O)[nH]c2=O)O[C@@H]1COP(O)(O)=S",


}

def estimate_percent_ionic_character_pauling(mol):
    # Pauling electronegativities for common elements
    pauling_en = {
        'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
        'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66,
        'Na': 0.93, 'K': 0.82, 'Mg': 1.31, 'Ca': 1.00, 'Zn': 1.65
    }

    total_ionic = 0.0
    bond_count = 0

    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        symbol1 = atom1.GetSymbol()
        symbol2 = atom2.GetSymbol()

        en1 = pauling_en.get(symbol1)
        en2 = pauling_en.get(symbol2)

        if en1 is not None and en2 is not None:
            delta_chi = abs(en1 - en2)
            percent_ionic = (1 - pow(2.71828, -0.25 * (delta_chi ** 2))) * 100
            total_ionic += percent_ionic
            bond_count += 1

    return round(total_ionic / bond_count, 2) if bond_count > 0 else 0.0


# Calculate properties
results = []

for name, smiles in smiles_dict.items():
    mol = Chem.MolFromSmiles(smiles)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    mw = Descriptors.MolWt(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    heavy_atoms = rdMolDescriptors.CalcNumHeavyAtoms(mol)
    flex_index = round(rot_bonds / heavy_atoms, 2) if heavy_atoms > 0 else 0
    ionic_percent = estimate_percent_ionic_character_pauling(mol)

    results.append({
        "Molecule": name,
        "LogP (Hydrophobicity)": round(logp, 2),
        "TPSA (Å²)": round(tpsa, 2),
        "Molecular Weight (g/mol)": round(mw, 2),
        "H-Bond Donors": hbd,
        "H-Bond Acceptors": hba,
	"Rotatable Bonds": rot_bonds,
	"Flexibility Index": flex_index,
	"Percent Ionic Character (%)": ionic_percent
    })

df = pd.DataFrame(results)
print(df)
