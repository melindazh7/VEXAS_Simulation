import pandas as pd

geneticData = pd.read_csv('metaDataExpLandau.csv')
geneticData.set_index('barcode', inplace=True)
gene_dict = geneticData.to_dict(orient = 'index')

additionalData = pd.read_csv('metaDataExpLandau_PPP1R15A_ATF4.csv')



import numpy as np
import pandas as pd

def merge_metadata_into_barcode_dict(cell_dict: dict, additionalData: pd.DataFrame,
                                      barcode_col: str = "barcode",
                                      fields_to_add: list = None) -> dict:
    """
    cell_dict: one group's existing barcode-keyed dict (e.g. Cornell_Code.muMye_dict)
    additionalData: DataFrame with a barcode column and the new fields
    fields_to_add: which columns to pull in, e.g. ["GADD34", "nCount_RNA", "nFeature_RNA"]

    Returns a NEW dict (does not mutate cell_dict in place) with the new
    fields merged into each cell's inner dict, matched by barcode.
    """
    if fields_to_add is None:
        fields_to_add = ["PPP1R15A", "nCount_RNA", "nFeature_RNA"]

    meta_indexed = additionalData.set_index(barcode_col)
    merged = {}
    n_matched, n_unmatched = 0, 0

    for barcode, inner_dict in cell_dict.items():
        new_inner = dict(inner_dict)  # shallow copy
        if barcode in meta_indexed.index:
            row = meta_indexed.loc[barcode]
            for field in fields_to_add:
                new_inner[field] = row[field] if field in row else np.nan
            n_matched += 1
        else:
            for field in fields_to_add:
                new_inner[field] = np.nan
            n_unmatched += 1
        merged[barcode] = new_inner

    return merged

muMye_dict = dict()
muLymph_dict = dict()
wtMye_dict = dict()
wtLymph_dict = dict()

for barcode, value in gene_dict.items():
    if value.get('Sample',0) !='PT06_A_CD34_positive' and value.get('Sample',0) != 'PT06_B_CD34_positive':
        if (value.get('Genotype_Controls') == "VEXAS_MUT") and (value.get('AllCellType') in ('GMP','BaEoMa','CD14 Mono','cDC')):
                muMye_dict.update({barcode:value})
        elif (value.get('Genotype_Controls') == "VEXAS_MUT") and (value.get('AllCellType') in ('B','CD4 T', 'CD8 T','CLP','NK','Plasma','LMPP')):
            muLymph_dict.update({barcode:value})
        elif (value.get('Genotype_Controls') == "VEXAS_WT") and (value.get('AllCellType') in ('GMP','BaEoMa','CD14 Mono','cDC')):
            wtMye_dict.update({barcode:value})
        elif (value.get('Genotype_Controls') == "VEXAS_WT") and (value.get('AllCellType') in ('B','CD4 T','CD8 T','CLP','NK','Plasma','LMPP')):
            wtLymph_dict.update({barcode:value})

muMye_dict_updated = merge_metadata_into_barcode_dict(muMye_dict, additionalData)
muLymph_dict_updated = merge_metadata_into_barcode_dict(muLymph_dict, additionalData)
wtMye_dict_updated = merge_metadata_into_barcode_dict(wtMye_dict, additionalData)
wtLymph_dict_updated = merge_metadata_into_barcode_dict(wtLymph_dict, additionalData)

#ATF4 max and min values
ATF4_muMyemax_bar = max(muMye_dict.values(), key=lambda x: x['ATF4'])
ATF4_muMyemax_value = ATF4_muMyemax_bar['ATF4']
non_zero_values = (ATF4 for sub_dict in muMye_dict.values() for AT4 in sub_dict.values() if val != 0)
atf4_values = [v['ATF4'] for v in muMye_dict.values() if v.get('ATF4') is not None]
nonzero_atf4 = [v for v in atf4_values if v != 0]
ATF4_muMyemin_value = min(nonzero_atf4) if nonzero_atf4 else None

ATF4_muLymphmax_bar = max(muLymph_dict.values(), key=lambda x: x['ATF4'])
ATF4_muLymphmax_value = ATF4_muLymphmax_bar['ATF4']
ATF4_muLymphmin_bar = min(muLymph_dict.values(), key=lambda x: x['ATF4'])
ATF4_muLymphmin_value = ATF4_muLymphmin_bar['ATF4']


ATF4_wtMyemax_bar = max(wtMye_dict.values(), key=lambda x: x['ATF4'])
ATF4_wtMyemax_value = ATF4_wtMyemax_bar['ATF4']
ATF4_wtMyemin_bar = min(wtMye_dict.values(), key=lambda x: x['ATF4'])
ATF4_wtMyemin_value = ATF4_wtMyemin_bar['ATF4']

ATF4_wtLymphmax_bar = max(wtLymph_dict.values(), key=lambda x: x['ATF4'])
ATF4_wtLymphmax_value = ATF4_wtLymphmax_bar['ATF4']
ATF4_wtLymphmin_bar = min(wtLymph_dict.values(), key=lambda x: x['ATF4'])
ATF4_wtLymphmin_value = ATF4_wtLymphmin_bar['ATF4']

#CHOP max and min values
CHOP_muMyemax_bar = max(muMye_dict.values(), key=lambda x: x['DDIT3'])
CHOP_muMyemax_value = CHOP_muMyemax_bar['DDIT3']
CHOP_muMye_values = [v['DDIT3'] for v in muMye_dict.values() if v.get('DDIT3') is not None]
CHOP_muMyemin_value = [v for v in CHOP_muMye_values if v != 0]
CHOP_muMyemin_value = min(CHOP_muMyemin_value) if CHOP_muMyemin_value else None

CHOP_muLymphmax_bar = max(muLymph_dict.values(), key=lambda x: x['DDIT3'])
CHOP_muLymphmax_value = CHOP_muLymphmax_bar['DDIT3']
CHOP_muLymph_values = [v['DDIT3'] for v in muLymph_dict.values() if v.get('DDIT3') is not None]
CHOP_muLymphmin_value = [v for v in CHOP_muLymph_values if v != 0]
CHOP_muLymphmin_value = min(CHOP_muLymphmin_value) if CHOP_muLymphmin_value else None

CHOP_wtMyemax_bar = max(wtMye_dict.values(), key=lambda x: x['DDIT3'])
CHOP_wtMyemax_value = CHOP_wtMyemax_bar['DDIT3']
CHOP_wtMye_values = [v['DDIT3'] for v in wtMye_dict.values() if v.get('DDIT3') is not None]
CHOP_wtMyemin_value = [v for v in CHOP_wtMye_values if v != 0]
CHOP_wtMyemin_value = min(CHOP_wtMyemin_value) if CHOP_wtMyemin_value else None

CHOP_wtLymphmax_bar = max(wtLymph_dict.values(), key=lambda x: x['DDIT3'])
CHOP_wtLymphmax_value = CHOP_wtLymphmax_bar['DDIT3']
CHOP_wtLymph_values = [v['DDIT3'] for v in wtLymph_dict.values() if v.get('DDIT3') is not None]
CHOP_wtLymphmin_value = [v for v in CHOP_wtLymph_values if v != 0]
CHOP_wtLymphmin_value = min(CHOP_wtLymphmin_value) if CHOP_wtLymphmin_value else None

#BAX max and min values
BAX_muMyemax_bar = max(muMye_dict.values(), key=lambda x: x['BAX'])
BAX_muMyemax_value = BAX_muMyemax_bar['BAX']
BAX_muMyemin_bar = min(muMye_dict.values(), key=lambda x: x['BAX'])
BAX_muMyemin_value = BAX_muMyemin_bar['BAX']

BAX_muLymphmax_bar = max(muLymph_dict.values(), key=lambda x: x['BAX'])
BAX_muLymphmax_value = BAX_muLymphmax_bar['BAX']
BAX_muLymphmin_bar = min(muLymph_dict.values(), key=lambda x: x['BAX'])
BAX_muLymphmin_value = BAX_muLymphmin_bar['BAX']

BAX_wtMyemax_bar = max(wtMye_dict.values(), key=lambda x: x['BAX'])
BAX_wtMyemax_value = BAX_wtMyemax_bar['BAX']
BAX_wtMyemin_bar = min(wtMye_dict.values(), key=lambda x: x['BAX'])
BAX_wtMyemin_value = BAX_wtMyemin_bar['BAX']

BAX_wtLymphmax_bar = max(wtLymph_dict.values(), key=lambda x: x['BAX'])
BAX_wtLymphmax_value = BAX_wtLymphmax_bar['BAX']
BAX_wtLymphmin_bar = min(wtLymph_dict.values(), key=lambda x: x['BAX'])
BAX_wtLymphmin_value = BAX_wtLymphmin_bar['BAX']

#BCL2 max and min values
BCL2_muMyemax_bar = max(muMye_dict.values(), key=lambda x: x['BCL2'])
BCL2_muMyemax_value = BCL2_muMyemax_bar['BCL2']
BCL2_muMyemin_bar = min(muMye_dict.values(), key=lambda x: x['BCL2'])
BCL2_muMyemin_value = BCL2_muMyemin_bar['BCL2']

BCL2_muLymphmax_bar = max(muLymph_dict.values(), key=lambda x: x['BCL2'])
BCL2_muLymphmax_value = BCL2_muLymphmax_bar['BCL2']
BCL2_muLymphmin_bar = min(muLymph_dict.values(), key=lambda x: x['BCL2'])
BCL2_muLymphmin_value = BCL2_muLymphmin_bar['BCL2']

BCL2_wtMyemax_bar = max(wtMye_dict.values(), key=lambda x: x['BCL2'])
BCL2wtMyemax_value = BCL2_wtMyemax_bar['BCL2']
BCL2_wtMyemin_bar = min(wtMye_dict.values(), key=lambda x: x['BCL2'])
BCL2_wtMyemin_value = BCL2_wtMyemin_bar['BCL2']

BCL2_wtLymphmax_bar = max(wtLymph_dict.values(), key=lambda x: x['BCL2'])
BCL2_wtLymphmax_value = BCL2_wtLymphmax_bar['BCL2']
BCL2_wtLymphmin_bar = min(wtLymph_dict.values(), key=lambda x: x['BCL2'])
BCL2_wtLymphmin_value = BCL2_wtLymphmin_bar['BCL2']

#muMye average values
ATF4_muMyeavg = 0
CHOP_muMyeavg = 0
BAX_muMyeavg = 0
BCL2_muMyeavg = 0

for barcode, value in muMye_dict.items():
    ATF4_muMyeavg += value.get('ATF4')
    CHOP_muMyeavg += value.get('DDIT3')
    BAX_muMyeavg += value.get('BAX')
    BCL2_muMyeavg += value.get('BCL2')

ATF4_muMyeavg = ATF4_muMyeavg / len(muMye_dict)
CHOP_muMyeavg = CHOP_muMyeavg / len(muMye_dict)
BAX_muMyeavg = BAX_muMyeavg / len(muMye_dict)
BCL2_muMyeavg = BCL2_muMyeavg / len(muMye_dict)

#muLymph average values
ATF4_muLymphavg = 0
CHOP_muLymphavg = 0
BAX_muLymphavg = 0
BCL2_muLymphavg = 0

for barcode, value in muLymph_dict.items():
    ATF4_muLymphavg += value.get('ATF4')
    CHOP_muLymphavg += value.get('DDIT3')
    BAX_muLymphavg += value.get('BAX')
    BCL2_muLymphavg += value.get('BCL2')

ATF4_muLymphavg = ATF4_muLymphavg / len(muLymph_dict)
CHOP_muLymphavg = CHOP_muLymphavg / len(muLymph_dict)
BAX_muLymphavg = BAX_muLymphavg / len(muLymph_dict)
BCL2_muLymphavg = BCL2_muLymphavg / len(muLymph_dict)

#wtMye average values
ATF4_wtMyeavg = 0
CHOP_wtMyeavg = 0
BAX_wtMyeavg = 0
BCL2_wtMyeavg = 0

for barcode, value in wtMye_dict.items():
    ATF4_wtMyeavg += value.get('ATF4')
    CHOP_wtMyeavg += value.get('DDIT3')
    BAX_wtMyeavg += value.get('BAX')
    BCL2_wtMyeavg += value.get('BCL2')

ATF4_wtMyeavg = ATF4_wtMyeavg / len(wtMye_dict)
CHOP_wtMyeavg = CHOP_wtMyeavg / len(wtMye_dict)
BAX_wtMyeavg = BAX_wtMyeavg / len(wtMye_dict)
BCL2_wtMyeavg = BCL2_wtMyeavg / len(wtMye_dict)

#wtLymph average values
ATF4_wtLymphavg = 0
CHOP_wtLymphavg = 0
BAX_wtLymphavg = 0
BCL2_wtLymphavg = 0

for barcode, value in wtLymph_dict.items():
    ATF4_wtLymphavg += value.get('ATF4')
    CHOP_wtLymphavg += value.get('DDIT3')
    BAX_wtLymphavg += value.get('BAX')
    BCL2_wtLymphavg += value.get('BCL2')

ATF4_wtLymphavg = ATF4_wtLymphavg / len(wtLymph_dict)
CHOP_wtLymphavg = CHOP_wtLymphavg / len(wtLymph_dict)
BAX_wtLymphavg = BAX_wtLymphavg / len(wtLymph_dict)
BCL2_wtLymphavg = BCL2_wtLymphavg / len(wtLymph_dict)

