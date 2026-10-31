#!/usr/bin/env python3
"""
Confronto Comparativo Diretto:
Ling Shao (2007) - Visual Hull Algebrico + PVC
vs.
De Sisternes et al. (Medical Physics 2015) - SMRD (Armoniche Sferiche + Deformazione Radiale)
"""

import os
import trimesh
import numpy as np
import matplotlib.pyplot as plt

def compare_models(patient_id="P_00232"):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dir_ling_shao = os.path.join(repo_root, "01_Baseline_Lesion_LingShao", "output", patient_id)
    dir_smrd = os.path.join(repo_root, "04_Spicular_Lesion_SMRD_Module", "output", patient_id)

    path_shao = os.path.join(dir_ling_shao, "meshes", "lesion_reconstructed.obj")
    if not os.path.exists(path_shao):
        cand = os.path.join(dir_ling_shao, f"{patient_id}_lesion.obj")
        if os.path.exists(cand):
            path_shao = cand

    path_smrd = os.path.join(dir_smrd, f"{patient_id}_smrd_lesion.obj")
    if not os.path.exists(path_smrd):
        cand = os.path.join(dir_smrd, "meshes", "lesion_reconstructed.obj")
        if os.path.exists(cand):
            path_smrd = cand

    if not os.path.exists(path_shao):
        print(f"[ERRORE] File non trovato: {path_shao}")
        return
    if not os.path.exists(path_smrd):
        print(f"[ERRORE] File non trovato: {path_smrd}")
        return

    mesh_shao = trimesh.load(path_shao)
    mesh_smrd = trimesh.load(path_smrd)

    vol_shao = abs(mesh_shao.volume)
    area_shao = mesh_shao.area
    psi_shao = (np.pi**(1/3) * (6 * vol_shao)**(2/3)) / area_shao

    vol_smrd = abs(mesh_smrd.volume)
    area_smrd = mesh_smrd.area
    psi_smrd = (np.pi**(1/3) * (6 * vol_smrd)**(2/3)) / area_smrd

    print(f"\n=== CONFRONTO GEOMETRICO ({patient_id}) ===")
    print(f"Ling Shao (2007): Volume={vol_shao:.1f} mm3, Area={area_shao:.1f} mm2, Sfericita={psi_shao:.3f}")
    print(f"SMRD (2015):      Volume={vol_smrd:.1f} mm3, Area={area_smrd:.1f} mm2, Sfericita={psi_smrd:.3f}")

    # Visualizzazione comparativa
    fig = plt.figure(figsize=(16, 8), facecolor='#0d1117')
    fig.suptitle(f"Confronto Algoritmi di Ricostruzione 3D Lesione Mammaria - {patient_id}",
                 fontsize=15, color='#58a6ff', fontweight='bold', y=0.97)

    # 1. Ling Shao - Isometria
    ax1 = fig.add_subplot(2, 4, 1, projection='3d', facecolor='#161b22')
    v_s = mesh_shao.vertices
    f_s = mesh_shao.faces
    ax1.plot_trisurf(v_s[:, 0], v_s[:, 1], v_s[:, 2], triangles=f_s, cmap='magma', edgecolor='none', alpha=0.9)
    ax1.set_title("Ling Shao (2007)\nVisual Hull Algebrico (3D)", color='#f85149', fontsize=11, fontweight='bold')
    ax1.view_init(elev=25, azim=45)
    ax1.tick_params(colors='#8b949e', labelsize=7)

    # 2. Ling Shao - Vista dall'alto (Axial) evidenzia le facce prismatiche a 45°
    ax2 = fig.add_subplot(2, 4, 2, projection='3d', facecolor='#161b22')
    ax2.plot_trisurf(v_s[:, 0], v_s[:, 1], v_s[:, 2], triangles=f_s, cmap='magma', edgecolor='none', alpha=0.9)
    ax2.set_title("Ling Shao - Vista Assiale (Top-Down)\n(Artefatto facce piane a 45°)", color='#f85149', fontsize=10)
    ax2.view_init(elev=90, azim=0)
    ax2.tick_params(colors='#8b949e', labelsize=7)

    # 3. SMRD (De Sisternes) - Isometria
    ax3 = fig.add_subplot(2, 4, 3, projection='3d', facecolor='#161b22')
    v_m = mesh_smrd.vertices
    f_m = mesh_smrd.faces
    ax3.plot_trisurf(v_m[:, 0], v_m[:, 1], v_m[:, 2], triangles=f_m, cmap='viridis', edgecolor='none', alpha=0.9)
    ax3.set_title("De Sisternes et al. (2015)\nSMRD Armoniche Sferiche (3D)", color='#3fb950', fontsize=11, fontweight='bold')
    ax3.view_init(elev=25, azim=45)
    ax3.tick_params(colors='#8b949e', labelsize=7)

    # 4. SMRD - Vista dall'alto (Axial) evidenzia la curvatura continua organica
    ax4 = fig.add_subplot(2, 4, 4, projection='3d', facecolor='#161b22')
    ax4.plot_trisurf(v_m[:, 0], v_m[:, 1], v_m[:, 2], triangles=f_m, cmap='viridis', edgecolor='none', alpha=0.9)
    ax4.set_title("De Sisternes - Vista Assiale\n(Curvatura organica continua)", color='#3fb950', fontsize=10)
    ax4.view_init(elev=90, azim=0)
    ax4.tick_params(colors='#8b949e', labelsize=7)

    # 5. Box descrittivo Ling Shao
    ax5 = fig.add_subplot(2, 2, 3, facecolor='#161b22')
    ax5.axis('off')
    txt_shao = (
        f"METODO 1: Ling Shao (2007) [Baseline Attuale]\n"
        f"--------------------------------------------------\n"
        f"Paradigma:     Visual Hull Algebrico + ART + PVC\n"
        f"Volume:        {vol_shao:.1f} mm3 ({vol_shao/1000:.2f} cm3)\n"
        f"Area Sup.:     {area_shao:.1f} mm2\n"
        f"Sfericita:     {psi_shao:.3f}\n"
        f"Caratteristica: Interseca cilindri estrusi a 45 gradi.\n"
        f"Criticità:     Tende ad assumere una sezione prismatica\n"
        f"               con spigoli vivi a 45 gradi nelle direzioni\n"
        f"               prive di raggi X (unconstrained angles)."
    )
    ax5.text(0.05, 0.5, txt_shao, color='#f85149', fontsize=10, family='monospace', verticalalignment='center')

    # 6. Box descrittivo De Sisternes
    ax6 = fig.add_subplot(2, 2, 4, facecolor='#161b22')
    ax6.axis('off')
    txt_smrd = (
        f"METODO 2: De Sisternes et al. (2015) [Nuovo SMRD]\n"
        f"--------------------------------------------------\n"
        f"Paradigma:     Spherical Harmonics + Radial Deformation\n"
        f"Volume:        {vol_smrd:.1f} mm3 ({vol_smrd/1000:.2f} cm3)\n"
        f"Area Sup.:     {area_smrd:.1f} mm2\n"
        f"Sfericita:     {psi_smrd:.3f} (+{((psi_smrd-psi_shao)/psi_shao)*100:.1f}% piu organica)\n"
        f"Caratteristica: Ottimizza una varieta continua 3D vincolata\n"
        f"               ai contorni radiologici reali di CC e MLO,\n"
        f"               minimizzando l'energia di bending (Laplace-Beltrami).\n"
        f"Vantaggio:     Zero spigoli prismatici, sfericita naturale\n"
        f"               e continuita differenziabile della lesione."
    )
    ax6.text(0.05, 0.5, txt_smrd, color='#3fb950', fontsize=10, family='monospace', verticalalignment='center')

    out_comp = os.path.join(os.path.dirname(dir_smrd), f"comparison_ling_shao_vs_smrd_{patient_id}.png")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_comp, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f" -> Immagine di confronto salvata: {out_comp}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Confronto comparativo Ling Shao (2007) vs SMRD (2015)")
    parser.add_argument('--patient', type=str, default=None, help="Paziente specifico da confrontare (es. P_00004 o P_00232)")
    args = parser.parse_args()

    if args.patient:
        compare_models(args.patient)
    else:
        for pid in ["P_00232", "P_01226", "P_01754"]:
            compare_models(pid)
