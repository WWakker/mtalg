import mtalg
import pytest

import mtalg.core.threads
from multiprocessing import cpu_count


class TestCore:
    def test1(self):
        with pytest.raises(ValueError):
            mtalg.set_num_threads(-1)

    def test2(self):
        with pytest.raises(ValueError):
            mtalg.set_num_threads(1.5)

    def test3(self, n_threads=4):
        assert mtalg.core.threads._global_num_threads == cpu_count()
        mtalg.set_num_threads(n_threads)
        assert mtalg.core.threads._global_num_threads == n_threads
        assert mtalg.get_num_threads() == n_threads
