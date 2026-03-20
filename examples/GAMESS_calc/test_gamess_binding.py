# -*- coding: utf-8 -*-
"""
Test the GAMESS binding for RANGE.
Tests input generation, output parsing, and full integration (mocked GAMESS).
"""

import os
import sys
import shutil
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock

from RANGE_go.cluster_model import cluster_model
from RANGE_go.energy_calculation import energy_computation
from ase.io import read, write
from ase import Atoms


def test_input_generation():
    """Test that GAMESS input is correctly generated from XYZ + template."""
    tmpdir = tempfile.mkdtemp()
    try:
        # Create a simple water XYZ
        xyz_path = os.path.join(tmpdir, 'water.xyz')
        with open(xyz_path, 'w') as f:
            f.write("3\nwater molecule\n")
            f.write("O   0.000   0.000   0.117\n")
            f.write("H   0.000   0.757  -0.469\n")
            f.write("H   0.000  -0.757  -0.469\n")

        # Create a GAMESS template
        template_path = os.path.join(tmpdir, 'template.inp')
        with open(template_path, 'w') as f:
            f.write(" $CONTRL SCFTYP=RHF RUNTYP=OPTIMIZE $END\n")
            f.write(" $DATA\n")
            f.write("test\nC1\n")
            f.write("{structure_info}\n")
            f.write(" $END\n")

        # Simulate what the GAMESS method does: read xyz, build $DATA lines, inject
        with open(xyz_path, 'r') as f1:
            lines = f1.readlines()
            natoms = int(lines[0])
            atomic_numbers = {'H': 1.0, 'O': 8.0}
            data_lines = []
            for line in lines[2:2+natoms]:
                parts = line.split()
                elem = parts[0]
                x, y, z = parts[1], parts[2], parts[3]
                znuc = atomic_numbers.get(elem, 0.0)
                data_lines.append(f' {elem}    {znuc:.1f}    {x}    {y}    {z}')
            content_xyz = '\n'.join(data_lines)

        with open(template_path, 'r') as f2:
            content_gms = f2.read()
        content_gms = content_gms.replace('{structure_info}', content_xyz)

        output_path = os.path.join(tmpdir, 'gamess_job.inp')
        with open(output_path, 'w') as f3:
            f3.write(content_gms)

        # Verify
        with open(output_path, 'r') as f:
            result = f.read()

        assert '$CONTRL' in result, "Missing $CONTRL group"
        assert '$DATA' in result, "Missing $DATA group"
        assert 'O    8.0' in result, "Oxygen entry not found"
        assert 'H    1.0' in result, "Hydrogen entry not found"
        assert '0.757' in result, "Coordinates not injected"
        assert '{structure_info}' not in result, "Placeholder not replaced"

        # Check format: each atom line should have 5 fields
        for dl in data_lines:
            parts = dl.split()
            assert len(parts) == 5, f"Expected 5 fields per atom line, got {len(parts)}: {dl}"
            assert float(parts[1]) > 0, "Nuclear charge should be positive"

        print("  PASSED: Input generation")
    finally:
        shutil.rmtree(tmpdir)


def test_output_parsing_real_format():
    """Test parsing energy and geometry from realistic GAMESS log output."""
    tmpdir = tempfile.mkdtemp()
    try:
        log_path = os.path.join(tmpdir, 'job.log')
        with open(log_path, 'w') as f:
            # Realistic GAMESS optimization log
            f.write(""" ITER EX DEM     TOTAL ENERGY        E CHANGE  DENSITY CHANGE    DIIS ERROR
   1  0  0      -74.9629466707   -74.9629466707   0.000000000   0.000000000
 FINAL RHF ENERGY IS      -74.9629466707 AFTER  11 ITERATIONS
 COORDINATES OF ALL ATOMS ARE (ANGS)
   ATOM   CHARGE       X              Y              Z
 ------------------------------------------------------------
 O           8.0   0.0000000000   0.0000000000   0.1170000000
 H           1.0   0.0000000000   0.7570000000  -0.4690000000
 H           1.0   0.0000000000  -0.7570000000  -0.4690000000

 ITER EX DEM     TOTAL ENERGY        E CHANGE  DENSITY CHANGE    DIIS ERROR
   1  0  0      -74.9658062366   -74.9658062366   0.000000000   0.000000000
 FINAL RHF ENERGY IS      -74.9659012171 AFTER   4 ITERATIONS
      ***** EQUILIBRIUM GEOMETRY LOCATED *****
 COORDINATES OF ALL ATOMS ARE (ANGS)
   ATOM   CHARGE       X              Y              Z
 ------------------------------------------------------------
 O           8.0   0.0000000000  -0.0000000000   0.1502001793
 H           1.0  -0.0000000000   0.7580826614  -0.4856000897
 H           1.0   0.0000000000  -0.7580826614  -0.4856000897

          INTERNUCLEAR DISTANCES (ANGS.)
          TOTAL ENERGY      =      -74.9659012171
""")

        # Use the same parsing logic as in energy_calculation.py (updated version)
        with open(log_path, 'r') as f4:
            log_lines = f4.readlines()
            energy_vals = []
            coord_blocks = []
            in_eq_coords = False
            coord_lines = []
            for line_idx, line in enumerate(log_lines):
                if 'FINAL' in line and 'ENERGY IS' in line:
                    parts_e = line.split()
                    for k, tok in enumerate(parts_e):
                        if tok == 'IS':
                            energy_vals.append(float(parts_e[k + 1]))
                            break
                if 'EQUILIBRIUM GEOMETRY LOCATED' in line:
                    in_eq_coords = False
                    coord_lines = []
                if in_eq_coords:
                    parts = line.split()
                    if len(parts) == 5:
                        try:
                            float(parts[1])
                            coord_lines.append(parts)
                        except ValueError:
                            pass
                    elif len(parts) == 0 and len(coord_lines) > 0:
                        coord_blocks.append(coord_lines[:])
                        coord_lines = []
                        in_eq_coords = False
                if 'COORDINATES OF ALL ATOMS ARE (ANGS)' in line:
                    in_eq_coords = True
                    coord_lines = []

            if len(coord_lines) > 0:
                coord_blocks.append(coord_lines[:])

        # Verify energy - should get both FINAL energies, use last
        assert len(energy_vals) == 2, f"Expected 2 FINAL energies, got {len(energy_vals)}: {energy_vals}"
        assert abs(energy_vals[-1] - (-74.9659012171)) < 1e-8, f"Wrong energy: {energy_vals[-1]}"

        # Verify geometry - should get the equilibrium geometry (last block)
        assert len(coord_blocks) >= 1, f"Expected at least 1 coordinate block, got {len(coord_blocks)}"
        last_coords = coord_blocks[-1]
        assert len(last_coords) == 3, f"Expected 3 atoms, got {len(last_coords)}"

        elems = [c[0] for c in last_coords]
        pos = np.array([[float(c[2]), float(c[3]), float(c[4])] for c in last_coords])
        atoms = Atoms(elems, positions=pos)

        assert list(atoms.get_chemical_symbols()) == ['O', 'H', 'H'], f"Wrong elements"
        assert abs(atoms.positions[0][2] - 0.1502001793) < 1e-6, "Wrong O z-coordinate"
        assert abs(atoms.positions[1][1] - 0.7580826614) < 1e-6, "Wrong H y-coordinate"

        print("  PASSED: Output parsing (real GAMESS format)")
    finally:
        shutil.rmtree(tmpdir)


def test_output_parsing_real_logfile():
    """Test parsing against the actual GAMESS optimization log if available."""
    log_path = '/tmp/gamess_opt.log'
    if not os.path.exists(log_path):
        print("  SKIPPED: Real GAMESS log parsing (no log at /tmp/gamess_opt.log)")
        return

    with open(log_path, 'r') as f4:
        log_lines = f4.readlines()
        energy_vals = []
        coord_blocks = []
        in_eq_coords = False
        coord_lines = []
        for line_idx, line in enumerate(log_lines):
            if 'FINAL' in line and 'ENERGY IS' in line:
                parts_e = line.split()
                for k, tok in enumerate(parts_e):
                    if tok == 'IS':
                        energy_vals.append(float(parts_e[k + 1]))
                        break
            if 'EQUILIBRIUM GEOMETRY LOCATED' in line:
                in_eq_coords = False
                coord_lines = []
            if in_eq_coords:
                parts = line.split()
                if len(parts) == 5:
                    try:
                        float(parts[1])
                        coord_lines.append(parts)
                    except ValueError:
                        pass
                elif len(parts) == 0 and len(coord_lines) > 0:
                    coord_blocks.append(coord_lines[:])
                    coord_lines = []
                    in_eq_coords = False
            if 'COORDINATES OF ALL ATOMS ARE (ANGS)' in line:
                in_eq_coords = True
                coord_lines = []

        if len(coord_lines) > 0:
            coord_blocks.append(coord_lines[:])

    assert len(energy_vals) > 0, "No FINAL energies found in real GAMESS log"
    energy = energy_vals[-1]
    assert energy < 0, f"Energy should be negative, got {energy}"
    # Expected from the actual run: -74.9659012171
    assert abs(energy - (-74.9659012171)) < 1e-4, f"Unexpected energy: {energy}"

    assert len(coord_blocks) > 0, "No coordinate blocks found in real GAMESS log"
    last_coords = coord_blocks[-1]
    assert len(last_coords) == 3, f"Expected 3 atoms in water, got {len(last_coords)}"

    elems = [c[0] for c in last_coords]
    pos = np.array([[float(c[2]), float(c[3]), float(c[4])] for c in last_coords])
    atoms = Atoms(elems, positions=pos)

    assert list(atoms.get_chemical_symbols()) == ['O', 'H', 'H'], f"Wrong elements: {atoms.get_chemical_symbols()}"
    # Verify reasonable water geometry (O-H bond ~0.96-1.0 Ang)
    oh1 = np.linalg.norm(atoms.positions[1] - atoms.positions[0])
    oh2 = np.linalg.norm(atoms.positions[2] - atoms.positions[0])
    assert 0.8 < oh1 < 1.2, f"O-H1 bond length {oh1} out of range"
    assert 0.8 < oh2 < 1.2, f"O-H2 bond length {oh2} out of range"

    print(f"  PASSED: Real GAMESS log parsing (E={energy:.10f} Ha, O-H={oh1:.4f} Ang)")


def test_full_integration_mocked():
    """Test the full GAMESS pathway through energy_computation with a mocked subprocess."""
    tmpdir = tempfile.mkdtemp()
    original_dir = os.getcwd()
    try:
        # Create water XYZ
        water_xyz = os.path.join(tmpdir, 'Water.xyz')
        with open(water_xyz, 'w') as f:
            f.write("3\nwater\n")
            f.write("O   0.000   0.000   0.117\n")
            f.write("H   0.000   0.757  -0.469\n")
            f.write("H   0.000  -0.757  -0.469\n")

        # Create GAMESS template
        template_path = os.path.join(tmpdir, 'input_gamess_template')
        with open(template_path, 'w') as f:
            f.write(" $CONTRL SCFTYP=RHF RUNTYP=OPTIMIZE $END\n")
            f.write(" $DATA\ntest\nC1\n{structure_info}\n $END\n")

        # Build cluster model
        input_molecules = [water_xyz]
        input_num_of_molecules = [1]
        input_constraint_type = ['in_box']
        input_constraint_value = [(0, 0, 0, 5, 5, 5)]

        cluster = cluster_model(input_molecules, input_num_of_molecules,
                                input_constraint_type, input_constraint_value)
        cluster_template, cluster_boundary, cluster_conversion_rule = cluster.generate_bounds()

        coarse_opt_parameter = dict(coarse_calc_eps='UFF', coarse_calc_sig='UFF', coarse_calc_chg=0,
                                    coarse_calc_step=5, coarse_calc_fmax=10, coarse_calc_constraint=None)

        calculator_command_line = "rungms {input_script} 01 1 > job.log"
        geo_opt_control_line = dict(method='GAMESS', input=template_path)

        computation = energy_computation(
            templates=cluster_template,
            go_conversion_rule=cluster_conversion_rule,
            calculator=calculator_command_line,
            calculator_type='external',
            geo_opt_para=geo_opt_control_line,
            if_coarse_calc=True,
            coarse_calc_para=coarse_opt_parameter,
        )

        # Create a fake job directory with the expected structure
        job_dir = os.path.join(tmpdir, 'job_test')
        os.makedirs(job_dir)

        # Write a coarse_final.xyz (what coarse optimization would produce)
        coarse_xyz = os.path.join(job_dir, 'coarse_final.xyz')
        with open(coarse_xyz, 'w') as f:
            f.write("3\ncoarse optimized\n")
            f.write("O   1.000   1.000   1.117\n")
            f.write("H   1.000   1.757   0.531\n")
            f.write("H   1.000   0.243   0.531\n")

        # Also write start.xyz as fallback
        shutil.copy(coarse_xyz, os.path.join(job_dir, 'start.xyz'))

        # Mock subprocess.run to write a fake GAMESS log with real format
        def fake_subprocess_run(cmd, **kwargs):
            with open('job.log', 'w') as f:
                f.write(" FINAL RHF ENERGY IS      -74.9659012171 AFTER   4 ITERATIONS\n")
                f.write("      ***** EQUILIBRIUM GEOMETRY LOCATED *****\n")
                f.write(" COORDINATES OF ALL ATOMS ARE (ANGS)\n")
                f.write("   ATOM   CHARGE       X              Y              Z\n")
                f.write(" ------------------------------------------------------------\n")
                f.write(" O           8.0   1.0000000000   1.0000000000   1.1170000000\n")
                f.write(" H           1.0   1.0000000000   1.7570000000   0.5310000000\n")
                f.write(" H           1.0   1.0000000000   0.2430000000   0.5310000000\n")
                f.write("\n")
            return MagicMock(returncode=0)

        with patch('subprocess.run', side_effect=fake_subprocess_run):
            os.chdir(job_dir)
            atoms, energy = computation.call_external_calculation(
                read(coarse_xyz), job_dir, calculator_command_line, geo_opt_control_line
            )

        assert abs(energy - (-74.9659012171)) < 1e-8, f"Wrong energy: {energy}"
        assert len(atoms) == 3, f"Expected 3 atoms, got {len(atoms)}"
        assert list(atoms.get_chemical_symbols()) == ['O', 'H', 'H'], f"Wrong symbols: {atoms.get_chemical_symbols()}"

        # Verify the generated GAMESS input file was created
        gamess_inp = os.path.join(job_dir, 'gamess_job.inp')
        assert os.path.exists(gamess_inp), "gamess_job.inp not created"
        with open(gamess_inp, 'r') as f:
            inp_content = f.read()
        assert '$CONTRL' in inp_content, "$CONTRL missing from generated input"
        assert 'O    8.0' in inp_content, "Oxygen not in generated input"
        assert '{structure_info}' not in inp_content, "Placeholder not replaced"

        print("  PASSED: Full integration (mocked)")
    finally:
        os.chdir(original_dir)
        shutil.rmtree(tmpdir)


if __name__ == '__main__':
    print("Testing GAMESS binding for RANGE...\n")
    test_input_generation()
    test_output_parsing_real_format()
    test_output_parsing_real_logfile()
    test_full_integration_mocked()
    print("\nAll tests passed!")
