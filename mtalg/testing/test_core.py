import mtalg
import pytest

import mtalg.core.threads


class TestCore:
    def test1(self):
        with pytest.raises(ValueError):
            mtalg.set_num_threads(-1)

    def test2(self):
        with pytest.raises(ValueError):
            mtalg.set_num_threads(1.5)

    def test3(self):
        assert mtalg.core.threads._global_num_threads is None
        mtalg.set_num_threads(4)
        assert mtalg.core.threads._global_num_threads == 4
