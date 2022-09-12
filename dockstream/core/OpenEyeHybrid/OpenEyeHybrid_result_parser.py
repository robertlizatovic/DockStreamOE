import pandas as pd
from openeye import oechem
from dockstream.core.result_parser import ResultParser

from dockstream.utils.enums.OE_Hybrid_enums import OpenEyeHybridOutputKeywordsEnum, OpenEyeHybridExecutablesEnum


class OpenEyeHybridResultParser(ResultParser):
    """Loads, parses and analyzes the output of an "OpenEye Hybrid" docking run, including poses and score."""
    def __init__(self, ligands: list, docking_mode: str):
        super().__init__(ligands=ligands)
        self._OE = OpenEyeHybridOutputKeywordsEnum()
        self._EE = OpenEyeHybridExecutablesEnum()
        if docking_mode == self._EE.HYBRID:
            self._scoring_func = self._OE.HYBRID_SCORE
        elif docking_mode == self._EE.FRED:
            self._scoring_func = self._OE.FRED_SCORE
        else:
            raise ValueError("Invalid docking mode used: {}".format(docking_mode))
        self._df_results = self._construct_dataframe()

    def _construct_dataframe(self) -> pd.DataFrame:
        def func_get_score(conformer):
            return float(oechem.OEGetSDData(conformer, self._scoring_func))

        return super()._construct_dataframe_with_funcobject(func_get_score)
