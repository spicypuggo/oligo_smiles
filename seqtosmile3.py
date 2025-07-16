from rdkit import Chem
from rdkit.Chem import AllChem
import re

# Define your monomers with attachment points as [*]
monomer_smiles = {
    "(mA)": "NC1=NC=NC2=C1N=CN2[C@]3([H])[C@](OC)([H])[C@]([*])([H])[C@@](COP([*])([OH])=O)([H])O3",
    "(mC)": "NC(C=CN1[C@@]2([H])[C@@](OC)([H])[C@@]([*])([H])[C@](COP([*])([OH])=O)([H])O2)=NC1=O",
    "(mG)": "O=C1C(N=CN2[C@@]3([H])[C@@](OC)([H])[C@@]([*])([H])[C@](COP([*])([OH])=O)([H])O3)=C2N=C(N)N1",
    "(mU)": "O=C(C=CN1[C@@]2([H])[C@@](OC)([H])[C@@]([*])([H])[C@](COP([*])([OH])=O)([H])O2)NC1=O",
    "(mA)#": "NC1=NC=NC2=C1N=CN2[C@@]3([H])[C@@](OC)([H])[C@@]([*])([H])[C@](COP([OH])([*])=S)([H])O3",
    "(mC)#": "NC(C=CN1[C@@]2([H])[C@@](OC)([H])[C@@]([*])([H])[C@](COP([*])([OH])=S)([H])O2)=NC1=O",
    "(mG)#": "O=C1C(N=CN2[C@@]3([H])[C@@](OC)([H])[C@@]([*])([H])[C@](COP([*])([OH])=S)([H])O3)=C2N=C(N)N1",
    "(mU)#": "O=C(C=CN1[C@@]2([H])[C@@](OC)([H])[C@@]([*])([H])[C@](COP([*])([OH])=S)([H])O2)NC1=O",
    "(fA)": "NC1=NC=NC2=C1N=CN2[C@@]3([H])[C@@](F)([H])[C@@]([*])([H])[C@](COP([*])([OH])=O)([H])O3",
    "(fC)": "NC(C=CN1[C@@]2([H])[C@@](F)([H])[C@@]([*])([H])[C@](COP([*])([OH])=O)([H])O2)=NC1=O",
    "(fG)": "O=C1C(N=CN2[C@@]3([H])[C@@](F)([H])[C@@]([*])([H])[C@](COP([OH])([*])=O)([H])O3)=C2N=C(N)N1",
    "(fU)": "O=C(C=CN1[C@@]2([H])[C@@](F)([H])[C@@]([*])([H])[C@](COP([*])([OH])=O)([H])O2)NC1=O",
    "(fA)#": "NC1=NC=NC2=C1N=CN2[C@@]3([H])[C@@](F)([H])[C@@]([*])([H])[C@](COP([*])([OH])=S)([H])O3",
    "(fC)#": "NC(C=CN1[C@@]2([H])[C@@](F)([H])[C@@]([*])([H])[C@](COP([*])([OH])=S)([H])O2)=NC1=O",
    "(fG)#": "O=C1C(N=CN2[C@@]3([H])[C@@](F)([H])[C@@]([*])([H])[C@](COP([OH])([*])=S)([H])O3)=C2N=C(N)N1",
    "(fU)#": "O=C(C=CN1[C@@]2([H])[C@@](F)([H])[C@@]([*])([H])[C@](COP([*])([OH])=S)([H])O2)NC1=O",
    "-Tegchol": "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)OC(=O)NCCCCC(CO)CO)C)C",
    "(DCA_Phos)": "OCC(COP([*])([OH])=O)CCCCNC(=O)CCCCCCCCCCCCCCCCCCCCC",
}

def remove_wildcards(mol):
    # Identify and remove atoms with atomic number 0 (wildcards like '*')
    wildcard = Chem.MolFromSmarts('[#0]')
    mol = Chem.DeleteSubstructs(mol, wildcard)
    Chem.SanitizeMol(mol)
    return mol

def connect_monomers(sequence__, monomer_smiles__):
    mols = []
    for i, base in enumerate(sequence__):
        smi = monomer_smiles__[base]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid input SMILES: {smi}")
        print(f"Monomer {i+1} cleaned SMILES: {Chem.MolToSmiles(mol)}")
        mols.append(mol)

    combined = mols[0]
    for mol in mols[1:]:
        combined = combine_monomers(combined, mol)

    return combined

def cap_5prime_with_OH(mol):
    """Replace the 5′ end phosphate-dummy connection with OH, preserving valence."""
    rw_mol = Chem.RWMol(mol)
    dummy_idx = None
    bridge_oxygen_idx = None
    phosphorus_idx = None
    negative_oxygen_idx = None

    for atom in rw_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:  # dummy [*]
            nbr = atom.GetNeighbors()[0]
            if nbr.GetAtomicNum() == 8:  # O
                for p in nbr.GetNeighbors():
                    if p.GetAtomicNum() == 15:
                        dummy_idx = atom.GetIdx()
                        bridge_oxygen_idx = nbr.GetIdx()
                        phosphorus_idx = p.GetIdx()
                        break
            elif nbr.GetAtomicNum() == 15:  # directly to P
                dummy_idx = atom.GetIdx()
                phosphorus_idx = nbr.GetIdx()
                break

    if dummy_idx is None or phosphorus_idx is None:
        print("Warning: 5′-end dummy not found.")
        return mol

    # Also find [O-] directly attached to P
    for nbr in rw_mol.GetAtomWithIdx(phosphorus_idx).GetNeighbors():
        if nbr.GetAtomicNum() == 8 and nbr.GetFormalCharge() == -1:
            negative_oxygen_idx = nbr.GetIdx()
            break

    # Remove atoms in reverse index order
    for idx in sorted(filter(lambda x: x is not None, [dummy_idx, bridge_oxygen_idx, negative_oxygen_idx]), reverse=True):
        rw_mol.RemoveAtom(idx)

    # Add OH (neutral O) to phosphorus
    new_O_idx = rw_mol.AddAtom(Chem.Atom(8))  # oxygen
    rw_mol.AddBond(phosphorus_idx, new_O_idx, Chem.BondType.SINGLE)

    mol = rw_mol.GetMol()
    Chem.SanitizeMol(mol)
    return mol

from rdkit import Chem

def cap_3prime_with_OH(mol):
    """Replaces the 3′ dummy atom with an OH group (–OH) on the ribose sugar."""
    rw_mol = Chem.RWMol(mol)

    dummy_idx = None
    carbon_idx = None

    # Find 3′-end dummy attached to carbon
    for atom in rw_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:  # Dummy atom
            neighbors = atom.GetNeighbors()
            if len(neighbors) != 1:
                continue
            neighbor = neighbors[0]
            if neighbor.GetAtomicNum() == 6:  # Carbon (typical for 3′ end)
                dummy_idx = atom.GetIdx()
                carbon_idx = neighbor.GetIdx()
                break

    if dummy_idx is None:
        print("Warning: 3′-end dummy not found.")
        return mol

    # Add OH: add O atom and H atom, and bond to carbon
    o_idx = rw_mol.AddAtom(Chem.Atom(8))  # Oxygen
    h_idx = rw_mol.AddAtom(Chem.Atom(1))  # Hydrogen
    rw_mol.AddBond(carbon_idx, o_idx, Chem.BondType.SINGLE)
    rw_mol.AddBond(o_idx, h_idx, Chem.BondType.SINGLE)

    # Remove dummy atom
    rw_mol.RemoveAtom(dummy_idx)

    mol = rw_mol.GetMol()
    Chem.SanitizeMol(mol)
    return mol

def find_attachment_points(mol):
    """Return a dict mapping '3prime' and '5prime' to dummy atom and its neighbor"""
    attachment_points = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:  # Dummy atom [*]
            neighbors = atom.GetNeighbors()
            if len(neighbors) != 1:
                raise ValueError("Dummy atom should have exactly one neighbor.")
            neighbor = neighbors[0]
            neighbor_idx = neighbor.GetIdx()
            atomic_num = neighbor.GetAtomicNum()

            print(f"Found dummy attached to atom {neighbor_idx} with atomic number {atomic_num}")

            if atomic_num == 15:
                # Directly attached to phosphorus
                attachment_points["5prime"] = (atom.GetIdx(), neighbor_idx)
            elif atomic_num == 8:
                # Oxygen: see if it's attached to a phosphorus
                for second_nbr in neighbor.GetNeighbors():
                    if second_nbr.GetAtomicNum() == 15:
                        attachment_points["5prime"] = (atom.GetIdx(), neighbor_idx)
                        break
                else:
                    # If not attached to P, treat as possible 3′
                    attachment_points["3prime"] = (atom.GetIdx(), neighbor_idx)
            elif atomic_num == 6:
                attachment_points["3prime"] = (atom.GetIdx(), neighbor_idx)
            else:
                raise ValueError(f"Unexpected neighbor atomic number: {atomic_num}")

    print("Attachment points found:", attachment_points)

    if "3prime" not in attachment_points or "5prime" not in attachment_points:
        raise ValueError("Could not identify both 3' and 5' attachment points.")

    return attachment_points

def combine_monomers(mol1, mol2):
    # Identify attachment points in each molecule
    attach1 = find_attachment_points(mol1)
    attach2 = find_attachment_points(mol2)

    d1, nbr1 = attach1["3prime"]      # from end of mol1
    d2, nbr2 = attach2["5prime"]      # start of mol2

    # Combine molecules
    combo = Chem.CombineMols(mol1, mol2)
    edcombo = Chem.EditableMol(combo)

    offset = mol1.GetNumAtoms()

    # Add bridging oxygen between 3′ C and 5′ P
    bridging_O_idx = edcombo.AddAtom(Chem.Atom(8))  # Oxygen atom
    edcombo.AddBond(nbr1, bridging_O_idx, Chem.BondType.SINGLE)  # C3′–O
    edcombo.AddBond(bridging_O_idx, nbr2 + offset, Chem.BondType.SINGLE)  # O–P

    # Remove dummy atoms
    to_remove = sorted([d1, d2 + offset], reverse=True)
    for idx in to_remove:
        edcombo.RemoveAtom(idx)

    return edcombo.GetMol()

# Define a sample 5-mer sequence using these monomers
s = "(mU)#(fA)#(mG)(fA)(fU)(fG)(mU)(fU)(mC)(fU)(mU)(fA)(mC)(fU)(mA)(fU)(mA)(mA)(mU)#(mU)#(fU)(DCA_Phos)"
result = re.findall(r'\([a-zA-Z]+\)#?', s)
sequence = (result)

# Step 1: Build the RNA molecule from monomers
rna_mol = connect_monomers(sequence, monomer_smiles)

# Step 2: Cap the 5′ end with OH
rna_mol = cap_5prime_with_OH(rna_mol)

# Step 3: Cap the 3′ end with OH
rna_mol = cap_3prime_with_OH(rna_mol)

# Step 3.1: Force RDKit to update stereochemistry flags
Chem.AssignStereochemistry(rna_mol, force=True, cleanIt=True)

# Step 3.2: Debug and visualize the molecule
from rdkit.Chem import Draw
Draw.MolToImage(rna_mol, size=(800, 800)).show()

# Step 4: Output SMILES
rna_smiles = Chem.MolToSmiles(rna_mol, isomericSmiles=True, canonical=True)
print("Assembled RNA SMILES (with 5′ and 3′ OH caps):\n", rna_smiles)

from rdkit.Chem import AllChem

# Ensure the molecule has hydrogens and a 3D conformation
rna_mol_with_H = Chem.AddHs(rna_mol)

# Step 5: Finalize the molecule and re-generate from the SMILES
rna_smiles = Chem.MolToSmiles(rna_mol, isomericSmiles=True, canonical=True)
rna_mol = Chem.MolFromSmiles(rna_smiles)

# Validate molecule
Chem.SanitizeMol(rna_mol)

rna_mol_with_H = Chem.AddHs(rna_mol)
Chem.SanitizeMol(rna_mol_with_H)

params = AllChem.ETKDG()
params.randomSeed = 0xf00d
params.useRandomCoords = True  # Try this for difficult molecules
AllChem.EmbedMolecule(rna_mol_with_H, params)
try:
    AllChem.MMFFOptimizeMolecule(rna_mol_with_H)
except:
    print("MMFF optimization failed. Falling back to UFF.")
    AllChem.UFFOptimizeMolecule(rna_mol_with_H)
print("Number of fragments:", len(Chem.GetMolFrags(rna_mol_with_H)))

# Try 3D embedding
result = AllChem.EmbedMolecule(rna_mol_with_H, params)
if result != 0:
    Chem.MolToMolFile(rna_mol_with_H, "failed_structure.mol")
    raise RuntimeError("Embedding failed. Molecule written to failed_structure.mol for inspection.")

# Optimize geometry
AllChem.UFFOptimizeMolecule(rna_mol_with_H)

# Save to .mol file
Chem.MolToMolFile(rna_mol_with_H, "assembled_rna.mol")

print("3D structure saved as 'assembled_rna.mol'. You can now open this file in Avogadro.")

