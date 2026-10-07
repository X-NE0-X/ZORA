from ENV_MGMT.imports import *







#----------------------------------------------------------------------------------------
class MarketData(pd.DataFrame):
    """
    ### What It Does
    Lightweight DataFrame proxy with OHLCV field selectors.

    #### Responsibility
    Exposes convenience properties such as `OPEN`, `HIGH`, `LOW`, `CLOSE`, and `VOLUME` while preserving normal DataFrame behavior.

    #### How To Use
    Use it as the data type returned by `CTX` when you want field-oriented column-selection helpers.

    #### Usage Example
    `obj = MarketData(...)`
    """

    @staticmethod
    def _select_field_columns(columns: Any, field: str) -> List[str]:

        field_lc = str(field).strip().lower()
        suffix = f"_{field_lc}"
        selected: List[str] = []

        for col in columns:

            if not isinstance(col, str):

                continue

            col_lc = col.lower()

            if col_lc == field_lc or col_lc.endswith(suffix):
                selected.append(col)


        return selected


    @property
    def _constructor(self) -> type["MarketData"]:

        return MarketData


    @property
    def OPEN(self) -> List[str]:
        """
        ### What It Does
        Handles OPEN behavior for `MarketData`.

        #### Responsibility
        Centralizes the validation, alignment, and dispatch rules used by `MarketData.OPEN`.

        #### How To Use
        Call `MarketData.OPEN(...)` from the workflow path that needs OPEN output.

        #### Usage Example
        `result = OPEN(...)`

        ---

        ### Returns
        - `result`: **List[str]**.
        """

        return self._select_field_columns(self.columns, "open")


    @property
    def HIGH(self) -> List[str]:
        """
        ### What It Does
        Handles HIGH behavior for `MarketData`.

        #### Responsibility
        Centralizes the validation, alignment, and dispatch rules used by `MarketData.HIGH`.

        #### How To Use
        Call `MarketData.HIGH(...)` from the workflow path that needs HIGH output.

        #### Usage Example
        `result = HIGH(...)`

        ---

        ### Returns
        - `result`: **List[str]**.
        """

        return self._select_field_columns(self.columns, "high")


    @property
    def LOW(self) -> List[str]:
        """
        ### What It Does
        Handles LOW behavior for `MarketData`.

        #### Responsibility
        Centralizes the validation, alignment, and dispatch rules used by `MarketData.LOW`.

        #### How To Use
        Call `MarketData.LOW(...)` from the workflow path that needs LOW output.

        #### Usage Example
        `result = LOW(...)`

        ---

        ### Returns
        - `result`: **List[str]**.
        """

        return self._select_field_columns(self.columns, "low")


    @property
    def CLOSE(self) -> List[str]:
        """
        ### What It Does
        Handles CLOSE behavior for `MarketData`.

        #### Responsibility
        Centralizes the validation, alignment, and dispatch rules used by `MarketData.CLOSE`.

        #### How To Use
        Call `MarketData.CLOSE(...)` from the workflow path that needs CLOSE output.

        #### Usage Example
        `result = CLOSE(...)`

        ---

        ### Returns
        - `result`: **List[str]**.
        """

        return self._select_field_columns(self.columns, "close")


    @property
    def VOLUME(self) -> List[str]:
        """
        ### What It Does
        Handles VOLUME behavior for `MarketData`.

        #### Responsibility
        Centralizes the validation, alignment, and dispatch rules used by `MarketData.VOLUME`.

        #### How To Use
        Call `MarketData.VOLUME(...)` from the workflow path that needs VOLUME output.

        #### Usage Example
        `result = VOLUME(...)`

        ---

        ### Returns
        - `result`: **List[str]**.
        """

        return self._select_field_columns(self.columns, "volume")
