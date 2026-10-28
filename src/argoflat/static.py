from typing import ClassVar

class VariablesShort:
    """A short list of Argo variables grouped by dimension."""

    LEVEL_VARS: ClassVar[tuple[str, ...]] = (
        'PRES',
        'TEMP',
        'PSAL',
    )

    PROF_VARS: ClassVar[tuple[str, ...]] = (
        'PLATFORM_NUMBER',
        'CYCLE_NUMBER',
        'JULD',
        'LATITUDE',
        'LONGITUDE',
    )

class Variables:
    """Argo variables grouped by dimension."""

    LEVEL_VARS: ClassVar[tuple[str, ...]] = (
        'PRES',
        'PRES_QC',
        'PRES_ADJUSTED',
        'PRES_ADJUSTED_QC',
        'PRES_ADJUSTED_ERROR',
        'TEMP',
        'TEMP_QC',
        'TEMP_ADJUSTED',
        'TEMP_ADJUSTED_QC',
        'TEMP_ADJUSTED_ERROR',
        'PSAL',
        'PSAL_QC',
        'PSAL_ADJUSTED',
        'PSAL_ADJUSTED_QC',
        'PSAL_ADJUSTED_ERROR',
    )

    PROF_VARS: ClassVar[tuple[str, ...]] = (
        'PLATFORM_NUMBER',
        'PROJECT_NAME',
        'PI_NAME',
        'CYCLE_NUMBER',
        'DIRECTION',
        'DATA_CENTRE',
        'DC_REFERENCE',
        'DATA_STATE_INDICATOR',
        'DATA_MODE',
        'PLATFORM_TYPE',
        'FLOAT_SERIAL_NO',
        'FIRMWARE_VERSION',
        'WMO_INST_TYPE',
        'JULD',
        'JULD_QC',
        'JULD_LOCATION',
        'LATITUDE',
        'LONGITUDE',
        'POSITION_QC',
        'POSITIONING_SYSTEM',
        'PROFILE_PRES_QC',
        'PROFILE_TEMP_QC',
        'PROFILE_PSAL_QC',
        'VERTICAL_SAMPLING_SCHEME',
        'CONFIG_MISSION_NUMBER',
    )