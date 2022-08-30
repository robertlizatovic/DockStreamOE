import tempfile
import os
import shutil
from copy import deepcopy
import multiprocessing
from enum import Enum
from typing import Optional, List, Any

from openeye import oechem, oequacpac
from pydantic import BaseModel
from typing_extensions import Literal

from dockstream.core.Schrodinger.Glide_docker import Parallelization
from dockstream.utils.enums.logging_enums import LoggingConfigEnum
from dockstream.utils.execute_external.OE_Hybrid import OpenEyeHybridExecutor

from dockstream.core.docker import Docker
from dockstream.core.OpenEyeHybrid.OpenEyeHybrid_result_parser import OpenEyeHybridResultParser
from dockstream.utils.execute_external.Omega import OmegaExecutor
from dockstream.utils.enums.Omega_enums import OmegaExecutablesEnum
from dockstream.utils.enums.OE_Hybrid_enums import OpenEyeHybridLigandPreparationEnum
from dockstream.utils.enums.OE_Hybrid_enums import OpenEyeHybridExecutablesEnum, OpenEyeHybridOutputKeywordsEnum
from dockstream.utils.general_utils import gen_temp_file

from dockstream.utils.translations.molecule_translator import MoleculeTranslator
from dockstream.utils.dockstream_exceptions import DockingRunFailed

_OM = OmegaExecutablesEnum()
_LP = OpenEyeHybridLigandPreparationEnum()
_EE = OpenEyeHybridExecutablesEnum()
_OE = OpenEyeHybridOutputKeywordsEnum()
_LE = LoggingConfigEnum()


class Resolution(str, Enum):
    HIGH = "High"              # 1000 poses passed
    STANDARD = "Standard"      # 100 poses passed
    LOW = "Low"


class OpenEyeHybridParameters(BaseModel):
    prefix_execution: Optional[str] = None
    binary_location: Optional[str] = None
    omega_prefix_execution: Optional[str] = None
    omega_binary_location : Optional[str] = None
    quacpac_correction: Optional[bool] = False
    omega_mode: Optional[str] = "pose"
    omega_gpu: Optional[str] = "false"
    docking_mode: Optional[str] = "fred"
    receptor_paths: Optional[List[str]] = None
    time_limit_per_compound: Optional[int] = None
    parallelization: Optional[Parallelization]
    resolution: Resolution = Resolution.STANDARD
    number_poses: int = 1

    def get(self, key: str) -> Any:
        """Temporary method to support nested_get"""
        return self.dict()[key]


class OpenEyeHybrid(Docker):
    """Interface to OpenEye Hybrid Backend"""

    backend: Literal["Hybrid"] = "Hybrid"
    parameters: OpenEyeHybridParameters

    _OpenEyeHybrid_executor: OpenEyeHybridExecutor() = None
    _omega_executor: OmegaExecutor() = None

    def __init__(self, **run_parameters):
        # invoke base class' constructor first
        super().__init__(**run_parameters)

        # prepare and check OpenEye Hybrid backend availability
        self._check_OpenEyeHybrid_backend_availability()
        self._check_Omega_backend_availability()

    def _check_OpenEyeHybrid_backend_availability(self):

        self._OpenEyeHybrid_executor = OpenEyeHybridExecutor(
            prefix_execution=self.parameters.prefix_execution,
            binary_location=self.parameters.binary_location
        )
        if not self._OpenEyeHybrid_executor.is_available(self.parameters.docking_mode):
            raise DockingRunFailed("Cannot initialize OpenEye {} docker, as OpenEye {} backend is not available - abort.".format(
                self.parameters.docking_mode, self.parameters.docking_mode))
        self._logger.log("Checked OpenEye {} backend availability (prefix_execution={}).".format(
            self.parameters.docking_mode, self.parameters.prefix_execution), _LE.DEBUG)

    def _check_Omega_backend_availability(self):
        self._omega_executor = OmegaExecutor(
            prefix_execution=self.parameters.omega_prefix_execution,
            binary_location=self.parameters.omega_binary_location
        )
        if not self._omega_executor.is_available():
            raise DockingRunFailed("Cannot initialize OpenEye Hybrid docker, as OpenEye Omega backend is not available - abort.")
        self._logger.log(f"Checked Omega backend availability (prefix_execution={self.parameters.omega_prefix_execution}).", _LE.DEBUG)
        

    def add_molecules(self, molecules: list):
        """This method overrides the parent class, docker.py add_molecules method. This method appends prepared
        ligands to a list for subsequent docking

        :param molecules: A list that is to contain all prepared ligands for subsequent docking
        :type molecules: list
        :raises NotImplementedError: Each backend must override the parent class, docker.py add_molecules method.
            Inability to do so or a bug causing incorrect implementation will raise a NotImplementedError
        """
        self._logger.log("Adding embedded molecules...", _LE.DEBUG)
        mol_trans = MoleculeTranslator(self.ligands, force_mol_type=_LP.TYPE_OPENEYE)
        mol_trans.add_molecules(molecules, bySMILES=False)
        self.ligands = mol_trans.get_as_openeye(bySMILES=False)
        self._logger.log("Finished adding molecules.", _LE.DEBUG)
        self._fix_ligands() # fix ligands prior to conformer expansion / docking
        self._docking_performed = False
    
    def _fix_ligands(self):
        """
        Uses QUACPAC to get a reasonable protomer (combination of tautomer and ionization state)
        of each ligand prior to generating conformers and performing docking. Important when these
        steps are not carried out by the ligand embedding stage.
        """
        if self.parameters.quacpac_correction not in [True, False]:
            raise ValueError(f"{self.parameters.quacpac_correction} flag is not valid. Supported values are:"
                             f"True, False")
        else:
            for ligand in self.ligands:
                lig_mol = ligand.get_molecule()
                if (lig_mol is not None) and self.parameters.quacpac_correction:
                    oequacpac.OEGetReasonableProtomer(lig_mol)

    def _generate_temporary_input_output_files(self, start_indices, sublists):
        # in case singletons are handed over, wrap them in a list for "zipping" later
        self._logger.log("Generating input files for OMEGA and Hybrid...", _LE.DEBUG)
        if not isinstance(start_indices, list):
            start_indices = [start_indices]
        if not isinstance(sublists, list):
            sublists = [sublists]

        tmp_output_dirs = []
        tmp_input_oeb_paths = []
        tmp_omega_oeb_paths = []
        tmp_output_sdf_paths = []
        for start_index, sublist in zip(start_indices, sublists):
            # generate temporary input files and output directory
            cur_tmp_output_dir = tempfile.mkdtemp()
            cur_tmp_oeb = gen_temp_file(prefix=str(start_index), suffix=".oeb", dir=cur_tmp_output_dir)
            cur_omega_oeb = gen_temp_file(prefix=str(start_index), suffix="_omega.oeb", dir=cur_tmp_output_dir)

            # write-out the temporary input file
            one_written = False
            ofs = oechem.oemolostream()
            if not ofs.open(cur_tmp_oeb):
                oechem.OEThrow.Fatal("Unable to open {} for writing".format(cur_tmp_oeb))
            for ligand in sublist:
                if ligand.get_molecule() is not None:
                    # set Title and all other SD tags
                    buffer = ligand.get_clone()
                    buffer.add_tags_to_molecule()
                    oechem.OEWriteMolecule(ofs, buffer.get_molecule())
                    one_written = True
            ofs.close()
            if not one_written:
                if os.path.isdir(cur_tmp_output_dir):
                    shutil.rmtree(cur_tmp_output_dir)
                continue

            tmp_output_dirs.append(cur_tmp_output_dir)
            tmp_input_oeb_paths.append(cur_tmp_oeb)
            tmp_omega_oeb_paths.append(cur_omega_oeb)

            # add the path to which "_dock_subjob()" will write the result SDF
            output_sdf_path = gen_temp_file(prefix=str(start_index), suffix="_result.sdf", dir=cur_tmp_output_dir)
            tmp_output_sdf_paths.append(output_sdf_path)
        return tmp_output_dirs, tmp_input_oeb_paths, tmp_omega_oeb_paths, tmp_output_sdf_paths

    def _dock(self, number_cores: int):
        # partition ligands into sublists and distribute to processor cores for docking
        start_indices, sublists = self.get_sublists_for_docking(number_cores=number_cores)
        number_sublists = len(sublists)
        self._logger.log(f"Split ligands into {number_sublists} sublists for docking.", _LE.DEBUG)
        sublists_submitted = 0
        slices_per_iteration = min(number_cores, number_sublists)

        while sublists_submitted < len(sublists):
            upper_bound_slice = min((sublists_submitted + slices_per_iteration), len(sublists))
            cur_slice_start_indices = start_indices[sublists_submitted:upper_bound_slice]
            cur_slice_sublists = sublists[sublists_submitted:upper_bound_slice]

            # generate paths and initialize molecules (so that if they fail, this can be covered)
            tmp_output_dirs, tmp_input_oeb_paths, \
            tmp_omega_oeb_paths, tmp_output_sdf_paths = self._generate_temporary_input_output_files(cur_slice_start_indices,
                                                                               cur_slice_sublists)

            # run in parallel; wait for all subjobs to finish before proceeding
            processes = []
            for chunk_index in range(len(tmp_output_dirs)):
                p = multiprocessing.Process(target=self._dock_subjob, args=(tmp_input_oeb_paths[chunk_index],
                                                                            tmp_omega_oeb_paths[chunk_index],
                                                                            tmp_output_sdf_paths[chunk_index],
                                                                            tmp_output_dirs[chunk_index]))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()

            # add the number of input sublists rather than the output temporary folders to account for cases where
            # entire sublists failed to produce an input structure
            sublists_submitted += len(cur_slice_sublists)

            # load the chunks and recombine the result; add conformations
            for chunk_index in range(len(tmp_output_dirs)):
                # this is a protection against the case where empty (file size == 0 bytes) files are generated due to
                # a failure during docking
                if not os.path.isfile(tmp_output_sdf_paths[chunk_index]) or os.path.getsize(tmp_output_sdf_paths[chunk_index]) == 0:
                    continue
                ifs = oechem.oemolistream()
                if not ifs.open(tmp_output_sdf_paths[chunk_index]):
                    oechem.OEThrow.Fatal("Unable to open %s for reading" % tmp_output_sdf_paths[chunk_index])
                # self._logger.log("Reading docked conformers from file {}".format(tmp_output_sdf_paths[chunk_index]), _LE.DEBUG)
                # need to read mols as single conformer OEGraphMol objects because SD data is only readily available for this OEMol type 
                for molecule in ifs.GetOEGraphMols():
                    conformer = oechem.OEGraphMol(molecule)
                    # if self.parameters.docking_mode == _EE.FRED:
                    #     dock_score = oechem.OEGetSDData(conformer, _OE.FRED_SCORE)
                    # else:
                    #     dock_score = oechem.OEGetSDData(conformer, _OE.HYBRID_SCORE)
                    cur_conformer_name = conformer.GetTitle().split("_")[0]
                    # self._logger.log("Read docked conformer {} with docking score {}".format(cur_conformer_name, dock_score), _LE.DEBUG)
                    # add molecule to the appropriate ligand
                    for ligand in self.ligands:
                        if ligand.get_identifier() == cur_conformer_name:
                            ligand.add_conformer(conformer)
                            break
                ifs.close()

            # clean-up
            for path in tmp_output_dirs:
                shutil.rmtree(path)
            self._log_docking_progress(number_done=sublists_submitted, number_total=number_sublists)

        # sort the conformers (best to worst), update their names to contain the conformer id and add tags
        # -> <ligand_number>:<enumeration>:<conformer_number>
        for ligand in self.ligands:
            ligand.set_conformers(sorted(ligand.get_conformers(),
                                         key=lambda x: self._get_score_from_conformer(conformer=x),
                                         reverse=False))
            ligand.add_tags_to_conformers()

        # log any docking fails
        self._docking_fail_check()

        # parse the result of the docking step
        result_parser = OpenEyeHybridResultParser(ligands=[ligand.get_clone() for ligand in self.ligands], 
            docking_mode=self.parameters.docking_mode)
        self._df_results = result_parser.as_dataframe()

        # set docking flag
        self._docking_performed = True
    
    def _prepare_omega_arguments(self) -> list:
        arguments_list = []

        # add the OMEGA mode specified
        if self.parameters.omega_mode not in [_OM.CLASSIC, _OM.MACROCYCLE, _OM.ROCS, _OM.POSE, _OM.DENSE]:
            raise ValueError(f"{self.parameters.omega_mode} mode is not valid. Supported modes are:"
                             f"'classic', 'macrocycle, 'rocs', 'pose', 'dense'")
        else:
            arguments_list.append(self.parameters.omega_mode)

        # Add other parameters to expose below
        # add the OMEGA GPU mode
        if self.parameters.omega_gpu not in ["true", "false"]:
            raise ValueError(f"{self.parameters.omega_gpu} flag is not valid. Supported flags are:"
                             f"'true', 'false'")
        else:
            arguments_list.extend([_OM.USE_GPU, self.parameters.omega_gpu])

        return arguments_list

    def _dock_subjob(self, input_oeb_path, omega_oeb_path, output_sdf_path, output_dir):

        # set up arguments list and execute
        # for an explanation of the parameters, see "OE_Hybrid_enums.py"
        # TODO: support "ensemble docking" - currently, only the first entry is used
        # run Omega here to generate the conformers
        # 1) prepare "OMEGA" arguments
        arguments = self._prepare_omega_arguments()
        arguments = arguments + [_OM.IN, input_oeb_path]
        arguments = arguments + [_OM.OUT, omega_oeb_path]

        # 2) run "OMEGA" backend and add log file to "debug" mode logging
        _ = self._omega_executor.execute(command=_OM.OMEGA,
                                              arguments=arguments,
                                              location=output_dir,
                                              check=False)
        self._logger.log(f"Executed OMEGA backend (output file: {input_oeb_path}).", _LE.DEBUG)
        for file in os.listdir(output_dir):
            if file.endswith(".fail"):
                with open(os.path.join(output_dir, file), "r") as f:
                    for line in f.readlines():
                        error_components = line.split(" ")
                        self._logger.log(f"It appears OMEGA failed to conformationally expand: {error_components[0]}, with ligand ID: {error_components[1]} "
                                         f", due to the following error: {' '.join(error_components[2:])}", _LE.DEBUG)
        # perform docking with conformationally expanded molecules
        oeb_paths = self.parameters.receptor_paths

        arguments = [_EE.RECEPTOR, oeb_paths[0],
                     _EE.DBASE, omega_oeb_path,
                     _EE.DOCKED_MOLECULE_FILE, output_sdf_path,
                    #  _EE.UNDOCKED_MOLECULES_FILE, output_dir,
                    #  _EE.SCORE_FILE, output_dir,
                    #  _EE.REPORT_FILE, output_dir,
                    #  _EE.SETTINGS_FILE, output_dir,
                    #  _EE.STATUS_FILE, output_dir,
                     _EE.DOCK_RESOLUTION, self.parameters.resolution.value,
                     _EE.NUM_POSES, self.parameters.number_poses,
                     _EE.NO_EXTRA_OUTPUT_FILES
                     ]

        execution_result = self._OpenEyeHybrid_executor.execute(command=self.parameters.docking_mode,
                                                                arguments=arguments,
                                                                check=False)
        self._delay4file_system(path=output_sdf_path)

        self._logger.log(f"Finished sublist (input: {input_oeb_path}, output directory: {output_dir}), with return code '{execution_result.returncode}'.", _LE.DEBUG)

    def write_docked_ligands(self, path, mode="all"):
        """This method overrides the parent class, docker.py write_docked_ligands method. This method writes docked
        ligands binding poses and conformers to a file. There is the option to output the best predicted binding pose
        per ligand, the best predicted binding pose per enumeration, or all the predicted binding poses

        :param path: Contains information on results output path
        :type path: string
        :param mode: Determines whether the output contains the best predicted binding pose per ligand, the best
            predicted binding pose per enumeration, or all the predicted binding poses
        :type mode: string, optional, default value is "all". Other possible values are "best_per_ligand" and
            "best_per_enumeration"
        :raises DockingRunFailed Error: This error is raised if the docking run has not been performed
        :raises OpenEye (OE) Fatal Error: This error is raised if the output file was unable to be created. Issues may
            be due to problems with the ligand structure
        :raises ValueError: This error is raised if the ligands are neither RDkit nor OpenEye readable
        """
        self._write_docked_ligands(path, mode, mol_type=_LP.TYPE_OPENEYE)

    def get_scores(self, best_only):
        """This method overrides the parent class, docker.py get_scores method. This method returns a list containing
        all docking scores. This method allows returning the best docking scores only. "best" can mean the minimum
        or maximum values for this given scoring function. By default, it will return the minimum values which is
        desired as Hybrid docking features fixed scoring/refinement functions with lowest scores being best. Returning
        all the docking scores (of different poses for instance) is also possible if best only is not enforced

        :param best_only: Determines whether the best (either minimum or maximum) docking scores are returned
        :type best_only: boolean, True or False
        :return: list of returned docking scores
        :raises ValueError: If best_only is True but neither "min" nor "max" was specified, a ValueError is raised
        """
        return self._get_scores(best_only=best_only, best="min")

    def write_result(self, path, mode="all", best="min"):
        """This method overrides the parent class, docker.py write_result method.
        This method writes the docking results to a csv file. There is the option to write out either the best
        predicted binding pose per ligand, the best predicted binding pose per enumeration or all the predicted
        binding poses

        :param path: Contains information on results output path
        :type path: string
        :param mode: Determines whether the output contains the best predicted binding pose per ligand, the best
            predicted binding pose per enumeration, or all the predicted binding poses
        :type mode: string, optional, default value is "all". Other possible value is "best_per_enumeration"
        :param best: Determines whether lower or higher values are better (typically lower ones)
        :type best: string, optional, default value is "min". Other possible value is "max"
        """
        return self._write_result(path=path, mode=mode, best="min")

    def _get_score_from_conformer(self, conformer):
        if self.parameters.docking_mode == _EE.HYBRID:
            return float(oechem.OEGetSDData(conformer, _OE.HYBRID_SCORE))
        elif self.parameters.docking_mode == _EE.FRED:
            return float(oechem.OEGetSDData(conformer, _OE.FRED_SCORE))
        else:
            return None

    def _sort_conformers(self, conformers: list, best=None) -> list:
        return super()._sort_conformers(conformers=conformers, best="min")

