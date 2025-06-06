"""Tests for nnls scipy.__version__ >= 1.12 - adapted from scipy tests."""

import numpy as np
import pytest
from numba import njit
from numpy.testing import assert_allclose

try:
    from scipy import __version__
    from scipy.optimize import nnls

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from numba_nnls import nnls_112_114

REASON = "test only for nnls in scipy.__version__ >= 1.12 if scipy is installed"


class TestNNLS:
    def setup_method(self):
        self.rng = np.random.default_rng(1685225766635251)

    def _test_nnls(self, func):
        a = np.arange(25.0).reshape(-1, 5)
        x = np.arange(5.0)
        y = a @ x
        x, res = func(a, y)
        assert res < 1e-7
        assert np.linalg.norm((a @ x) - y) < 1e-7

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_nnls_112_114(self):
        self._test_nnls(nnls_112_114)

    @pytest.mark.skipif(not HAS_SCIPY or __version__ < "1.12", reason=REASON)
    def test_nnls_njit(self):
        self._test_nnls(njit(lambda a, b: nnls(a, b)))

    def _test_nnls_tall(self, func):
        a = self.rng.uniform(low=-10, high=10, size=[50, 10])
        x = np.abs(self.rng.uniform(low=-2, high=2, size=[10]))
        x[::2] = 0
        b = a @ x
        xact, rnorm = func(a, b, atol=500 * np.linalg.norm(a, 1) * np.spacing(1.0))
        assert_allclose(xact, x, rtol=0.0, atol=1e-10)
        assert rnorm < 1e-12

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_nnls_112_114_tall(self):
        self._test_nnls_tall(nnls_112_114)

    @pytest.mark.skipif(not HAS_SCIPY or __version__ < "1.12", reason=REASON)
    def test_nnls_njit_tall(self):
        self._test_nnls_tall(njit(lambda a, b, atol: nnls(a, b, atol=atol)))

    def _test_nnls_wide(self, func):
        # If too wide then problem becomes too ill-conditioned ans starts
        # emitting warnings, hence small m, n difference.
        a = self.rng.uniform(low=-10, high=10, size=[100, 120])
        x = np.abs(self.rng.uniform(low=-2, high=2, size=[120]))
        x[::2] = 0
        b = a @ x
        xact, rnorm = func(a, b, atol=500 * np.linalg.norm(a, 1) * np.spacing(1.0))
        assert_allclose(xact, x, rtol=0.0, atol=1e-10)
        assert rnorm < 1e-12

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_nnls_112_114_wide(self):
        self._test_nnls_wide(nnls_112_114)

    @pytest.mark.skipif(not HAS_SCIPY or __version__ < "1.12", reason=REASON)
    def test_nnls_njit_wide(self):
        self._test_nnls_wide(njit(lambda a, b, atol: nnls(a, b, atol=atol)))

    def _test_maxiter(self, func):
        # test that maxiter argument does stop iterations
        a = self.rng.uniform(size=(5, 10))
        b = self.rng.uniform(size=5)
        with pytest.raises(RuntimeError):
            func(a, b, maxiter=1)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_maxiter_new(self):
        self._test_maxiter(nnls_112_114)

    @pytest.mark.skipif(not HAS_SCIPY or __version__ < "1.12", reason=REASON)
    def test_maxiter_njit(self):
        self._test_maxiter(njit(lambda a, b, maxiter: nnls(a, b, maxiter=maxiter)))

    def _test_nnls_inner_loop_case1(self, func):
        # See scipy gh-20168
        n = np.array(
            [
                3,
                2,
                0,
                1,
                1,
                1,
                3,
                8,
                14,
                16,
                29,
                23,
                41,
                47,
                53,
                57,
                67,
                76,
                103,
                89,
                97,
                94,
                85,
                95,
                78,
                78,
                78,
                77,
                73,
                50,
                50,
                56,
                68,
                98,
                95,
                112,
                134,
                145,
                158,
                172,
                213,
                234,
                222,
                215,
                216,
                216,
                206,
                183,
                135,
                156,
                110,
                92,
                63,
                60,
                52,
                29,
                20,
                16,
                12,
                5,
                5,
                5,
                1,
                2,
                3,
                0,
                2,
            ]
        )
        k = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.7205812007860187,
                0.0,
                1.4411624015720375,
                0.7205812007860187,
                2.882324803144075,
                5.76464960628815,
                5.76464960628815,
                12.249880413362318,
                15.132205216506394,
                20.176273622008523,
                27.382085629868712,
                48.27894045266326,
                47.558359251877235,
                68.45521407467177,
                97.99904330689854,
                108.0871801179028,
                135.46926574777152,
                140.51333415327366,
                184.4687874012208,
                171.49832578707245,
                205.36564222401535,
                244.27702706646033,
                214.01261663344755,
                228.42424064916793,
                232.02714665309804,
                205.36564222401535,
                172.9394881886445,
                191.67459940908097,
                162.1307701768542,
                153.48379576742198,
                110.96950492104689,
                103.04311171240067,
                86.46974409432225,
                60.528820866025576,
                43.234872047161126,
                23.779179625938617,
                24.499760826724636,
                17.29394881886445,
                11.5292992125763,
                5.76464960628815,
                5.044068405502131,
                3.6029060039300935,
                0.0,
                2.882324803144075,
                0.0,
                0.0,
                0.0,
            ]
        )
        d = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.003889242101538,
                0.0,
                0.007606268390096,
                0.0,
                0.025457371599973,
                0.036952882091577,
                0.0,
                0.08518359183449,
                0.048201126400243,
                0.196234990022205,
                0.144116240157247,
                0.171145134062442,
                0.0,
                0.0,
                0.269555036538714,
                0.0,
                0.0,
                0.0,
                0.010893241091872,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.048167058272886,
                0.011238724891049,
                0.0,
                0.0,
                0.055162603456078,
                0.0,
                0.0,
                0.0,
                0.0,
                0.027753339088588,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        # The following code sets up a system of equations such that
        # $k_i-p_i*n_i$ is minimized for $p_i$ with weights $n_i$ and
        # monotonicity constraints on $p_i$. This translates to a system of
        # equations of the form $k_i - (d_1 + ... + d_i) * n_i$ and
        # non-negativity constraints on the $d_i$. If $n_i$ is zero the
        # system is modified such that $d_i - d_{i+1}$ is then minimized.
        N = len(n)
        A = np.diag(n) @ np.tril(np.ones((N, N)))
        w = n**0.5

        nz = (n == 0).nonzero()[0]
        A[nz, nz] = 1
        A[nz, np.minimum(nz + 1, N - 1)] = -1
        w[nz] = 1
        k[nz] = 0
        W = np.diag(w)

        # Small perturbations can already make the infinite loop go away (just
        # uncomment the next line)
        # k = k + 1e-10 * np.random.normal(size=N)
        dact, _ = func(W @ A, W @ k)
        assert_allclose(dact, d, rtol=0.0, atol=1e-10)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_nnls_112_114_inner_loop_case1(self):
        self._test_nnls_inner_loop_case1(nnls_112_114)

    @pytest.mark.skipif(not HAS_SCIPY or __version__ < "1.12", reason=REASON)
    def test_nnls_njit_inner_loop_case1(self):
        self._test_nnls_inner_loop_case1(njit(lambda a, b: nnls(a, b)))

    def _test_nnls_inner_loop_case2(self, func):
        # See scipy gh-20168
        n = np.array(
            [
                1,
                0,
                1,
                2,
                2,
                2,
                3,
                3,
                5,
                4,
                14,
                14,
                19,
                26,
                36,
                42,
                36,
                64,
                64,
                64,
                81,
                85,
                85,
                95,
                95,
                95,
                75,
                76,
                69,
                81,
                62,
                59,
                68,
                64,
                71,
                67,
                74,
                78,
                118,
                135,
                153,
                159,
                210,
                195,
                218,
                243,
                236,
                215,
                196,
                175,
                185,
                149,
                144,
                103,
                104,
                75,
                56,
                40,
                32,
                26,
                17,
                9,
                12,
                8,
                2,
                1,
                1,
                1,
            ]
        )
        k = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.7064355064917867,
                0.0,
                0.0,
                2.11930651947536,
                0.7064355064917867,
                0.0,
                3.5321775324589333,
                7.064355064917867,
                11.302968103868587,
                16.95445215580288,
                20.486629688261814,
                20.486629688261814,
                37.44108184406469,
                55.808405012851146,
                78.41434122058831,
                103.13958394780086,
                105.965325973768,
                125.74552015553803,
                149.057891869767,
                176.60887662294667,
                197.09550631120848,
                211.930651947536,
                204.86629688261814,
                233.8301526487814,
                221.1143135319292,
                195.6826352982249,
                197.80194181770025,
                191.4440222592742,
                187.91184472681525,
                144.11284332432447,
                131.39700420747232,
                116.5618585711448,
                93.24948685691584,
                89.01087381796512,
                53.68909849337579,
                45.211872415474346,
                31.083162285638615,
                24.72524272721253,
                16.95445215580288,
                9.890097090885014,
                9.890097090885014,
                2.8257420259671466,
                2.8257420259671466,
                1.4128710129835733,
                0.7064355064917867,
                1.4128710129835733,
            ]
        )
        d = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0021916146355674473,
                0.0,
                0.0,
                0.011252740799789484,
                0.0,
                0.0,
                0.037746623295934395,
                0.03602328132946222,
                0.09509167709829734,
                0.10505765870204821,
                0.01391037014274718,
                0.0188296228752321,
                0.20723559202324254,
                0.3056220879462608,
                0.13304643490426477,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.043185876949706214,
                0.0037266261379722554,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.094797899357143,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.23450935613672663,
                0.0,
                0.0,
                0.07064355064917871,
            ]
        )
        # The following code sets up a system of equations such that
        # $k_i-p_i*n_i$ is minimized for $p_i$ with weights $n_i$ and
        # monotonicity constraints on $p_i$. This translates to a system of
        # equations of the form $k_i - (d_1 + ... + d_i) * n_i$ and
        # non-negativity constraints on the $d_i$. If $n_i$ is zero the
        # system is modified such that $d_i - d_{i+1}$ is then minimized.
        N = len(n)
        A = np.diag(n) @ np.tril(np.ones((N, N)))
        w = n**0.5

        nz = (n == 0).nonzero()[0]
        A[nz, nz] = 1
        A[nz, np.minimum(nz + 1, N - 1)] = -1
        w[nz] = 1
        k[nz] = 0
        W = np.diag(w)

        dact, _ = func(W @ A, W @ k, atol=1e-7)

        p = np.cumsum(dact)
        assert np.all(dact >= 0)
        assert np.linalg.norm(k - n * p, ord=np.inf) < 28
        assert_allclose(dact, d, rtol=0.0, atol=1e-10)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_nnls_112_114_inner_loop_case2(self):
        self._test_nnls_inner_loop_case2(nnls_112_114)

    @pytest.mark.skipif(not HAS_SCIPY or __version__ < "1.12", reason=REASON)
    def test_nnls_njit_inner_loop_case2(self):
        self._test_nnls_inner_loop_case2(njit(lambda a, b, atol: nnls(a, b, atol=atol)))

    def _test_nnls_gh20302(self, func):
        # See gh-20302
        A = np.array(
            [
                0.33408569134321575,
                0.11136189711440525,
                0.049140798007949286,
                0.03712063237146841,
                0.055680948557202625,
                0.16642814595936478,
                0.11095209730624318,
                0.09791993030943345,
                0.14793612974165757,
                0.44380838922497273,
                0.11099502671044059,
                0.11099502671044059,
                0.14693672599330593,
                0.3329850801313218,
                1.498432860590948,
                0.0832374225132955,
                0.11098323001772734,
                0.19589481249472837,
                0.5919105600945457,
                3.5514633605672747,
                0.06658716751427037,
                0.11097861252378394,
                0.24485832778293645,
                0.9248217710315328,
                6.936163282736496,
                0.05547609388181014,
                0.11095218776362029,
                0.29376003042571264,
                1.3314262531634435,
                11.982836278470993,
                0.047506113282944136,
                0.11084759766020298,
                0.3423969672933396,
                1.8105107617833156,
                19.010362998724812,
                0.041507335004505576,
                0.11068622667868154,
                0.39074115283013344,
                2.361306169145206,
                28.335674029742474,
                0.03682846280947718,
                0.11048538842843154,
                0.4387861797121048,
                2.9831054875676517,
                40.2719240821633,
                0.03311278164362387,
                0.11037593881207958,
                0.4870572300443105,
                3.6791979604026523,
                55.187969406039784,
                0.030079304092299915,
                0.11029078167176636,
                0.5353496017200152,
                4.448394860761242,
                73.3985152025605,
                0.02545939709595835,
                0.11032405408248619,
                0.6328767609778363,
                6.214921713313388,
                121.19097340961108,
                0.022080881724881523,
                0.11040440862440762,
                0.7307742886903428,
                8.28033064683057,
                186.30743955368786,
                0.020715838214945492,
                0.1104844704797093,
                0.7800578384588346,
                9.42800814760186,
                226.27219554244465,
                0.01843179728340054,
                0.11059078370040323,
                0.8784095015912599,
                11.94380463964355,
                322.48272527037585,
                0.015812787653789077,
                0.11068951357652354,
                1.0257259848595766,
                16.27135849574896,
                512.5477926160922,
                0.014438550529330062,
                0.11069555405819713,
                1.1234754801775881,
                19.519316032262093,
                673.4164031130423,
                0.012760770585072577,
                0.110593345070629,
                1.2688431112524712,
                24.920367089248398,
                971.8943164806875,
                0.011427556646114315,
                0.11046638091243838,
                1.413623342459821,
                30.967408782453557,
                1347.0822820367298,
                0.010033330264470307,
                0.11036663290917338,
                1.6071533470570285,
                40.063087746029936,
                1983.122843428482,
                0.008950061496507258,
                0.11038409179025618,
                1.802244865119193,
                50.37194055362024,
                2795.642700725923,
                0.008071078821135658,
                0.11030474388885401,
                1.9956465761433504,
                61.80742482572119,
                3801.1566267818534,
                0.007191031207777556,
                0.11026247851925586,
                2.238160187262168,
                77.7718015155818,
                5366.2543045751445,
                0.00636834224248,
                0.11038459886965334,
                2.5328963107984297,
                99.49331844784753,
                7760.4788389321075,
                0.005624259098118485,
                0.11061042892966355,
                2.879742607664547,
                128.34496770138628,
                11358.529641572684,
                0.0050354270614989555,
                0.11077939535297703,
                3.2263279459292575,
                160.85168205252265,
                15924.316523199741,
                0.0044997853165982555,
                0.1109947044760903,
                3.6244287189055613,
                202.60233390369015,
                22488.859063309606,
                0.004023601950058174,
                0.1113196539516095,
                4.07713905729421,
                255.6270320242126,
                31825.565487014468,
                0.0036024117873727094,
                0.111674765408554,
                4.582933773135057,
                321.9583486728612,
                44913.18963986413,
                0.003201503089582304,
                0.11205260813538065,
                5.191786833370116,
                411.79333489752383,
                64857.45024636,
                0.0028633044552448853,
                0.11262330857296549,
                5.864295861648949,
                522.7223161899905,
                92521.84996562831,
                0.0025691897303891965,
                0.11304434813712465,
                6.584584405106342,
                656.5615739804199,
                129999.19164812315,
                0.0022992911894424675,
                0.11343169867916175,
                7.4080129906658305,
                828.2026426227864,
                183860.98666225857,
                0.0020449922071108764,
                0.11383789952917212,
                8.388975556433872,
                1058.2750599896935,
                265097.9025274183,
                0.001831274615120854,
                0.11414945100919989,
                9.419351803810935,
                1330.564050780237,
                373223.2162438565,
                0.0016363333454631633,
                0.11454333418242145,
                10.6143816579462,
                1683.787012481595,
                530392.9089317025,
                0.0014598610433380044,
                0.11484240207592301,
                11.959688127956882,
                2132.0874753402027,
                754758.9662704318,
                0.0012985240015312626,
                0.11513579480243862,
                13.514425358573531,
                2715.5160990137824,
                1083490.9235064993,
                0.0011614735761289934,
                0.11537304189548002,
                15.171418602667567,
                3415.195870828736,
                1526592.554260445,
                0.0010347472698811352,
                0.11554677847006009,
                17.080800985009617,
                4322.412404600832,
                2172012.2333119176,
                0.0009232988811258664,
                0.1157201264344419,
                19.20004861829407,
                5453.349531598553,
                3075689.135821584,
                0.0008228871862975205,
                0.11602709326795038,
                21.65735242414206,
                6920.203923780365,
                4390869.389638642,
                0.00073528900066722,
                0.11642075843897651,
                24.40223571298994,
                8755.811207598026,
                6238515.485413593,
                0.0006602764384729194,
                0.11752920604817965,
                27.694443541914293,
                11171.386093291572,
                8948280.260726549,
                0.0005935538977939806,
                0.11851292825953147,
                31.325508920763063,
                14174.185724149384,
                12735505.873148222,
                0.0005310755355633124,
                0.11913794514470308,
                35.381052949627765,
                17987.010118815077,
                18157886.71494382,
                0.00047239949671590953,
                0.1190446731724092,
                39.71342528048061,
                22679.438775422022,
                25718483.571328573,
                0.00041829129789387623,
                0.11851586773659825,
                44.45299332965028,
                28542.57147989741,
                36391778.63686921,
                0.00037321512015419886,
                0.11880681324908665,
                50.0668539579632,
                36118.26128449941,
                51739409.29004541,
                0.0003315539616702064,
                0.1184752823034871,
                56.04387059062639,
                45383.29960621684,
                72976345.76679668,
                0.00029456064937920213,
                0.11831519416731286,
                62.91195073220101,
                57265.53993693082,
                103507463.43600245,
                0.00026301867496859703,
                0.11862142241083726,
                70.8217262087034,
                72383.14781936012,
                146901598.49939138,
                0.00023618734450420032,
                0.11966825454879482,
                80.26535457124461,
                92160.51176984518,
                210125966.835247,
                0.00021165918071578316,
                0.12043407382728061,
                90.7169587544247,
                116975.56852918258,
                299515943.218972,
                0.00018757727511329545,
                0.11992440455576689,
                101.49899864101785,
                147056.26174166967,
                423080865.0307836,
                0.00016654469159895833,
                0.11957908856805206,
                113.65970431102812,
                184937.67016486943,
                597533612.3026931,
                0.00014717439179415048,
                0.11872067604728138,
                126.77899683346702,
                231758.58906776624,
                841283678.3159915,
                0.00012868496382376066,
                0.1166314722122684,
                139.93635237349534,
                287417.30847929465,
                1172231492.6328032,
                0.00011225559452625302,
                0.11427619522772557,
                154.0034283704458,
                355281.4912295324,
                1627544511.322488,
                9.879511142981067e-05,
                0.11295574406808354,
                170.96532050841535,
                442971.0111288653,
                2279085852.2580123,
                8.71257780313587e-05,
                0.11192758284428547,
                190.35067416684697,
                554165.2523674504,
                3203629323.93623,
                7.665069027765277e-05,
                0.11060694607065294,
                211.28835951100046,
                690933.608546013,
                4486577387.093535,
                6.734021094824451e-05,
                0.10915848194710433,
                234.24338803525194,
                860487.9079859136,
                6276829044.8032465,
                5.9191625040287665e-05,
                0.10776821865668373,
                259.7454711820425,
                1071699.0387579766,
                8780430224.544102,
                5.1856803674907676e-05,
                0.10606444911641115,
                287.1843540288165,
                1331126.3723998806,
                12251687131.5685,
                4.503421404759231e-05,
                0.10347361247668461,
                314.7338642485931,
                1638796.0697522392,
                16944331963.203278,
                3.90470387455642e-05,
                0.1007804070023012,
                344.3427560918527,
                2014064.4865519698,
                23392351979.057854,
                3.46557661636393e-05,
                0.10046706610839032,
                385.56603915081587,
                2533036.2523656,
                33044724430.235435,
                3.148745865254635e-05,
                0.1025441570117926,
                442.09038234164746,
                3262712.3882769793,
                47815050050.199135,
                2.9790762078715404e-05,
                0.1089845379379672,
                527.8068231298969,
                4375751.903321453,
                72035815708.42941,
                2.8772639817606534e-05,
                0.11823636789048445,
                643.2048194503195,
                5989838.001888927,
                110764084330.93005,
                2.7951691815106586e-05,
                0.12903432664913705,
                788.5500418523591,
                8249371.000613411,
                171368308481.2427,
                2.6844392423114212e-05,
                0.1392060709754626,
                955.6296403631383,
                11230229.319931043,
                262063016295.25085,
                2.499458273851386e-05,
                0.14559344445184325,
                1122.7022399726002,
                14820229.698461473,
                388475270970.9214,
                2.337386729019776e-05,
                0.15294300496886065,
                1324.8158105672455,
                19644861.137128454,
                578442936182.7473,
                2.0081014872174113e-05,
                0.14760215298210377,
                1436.2385042492353,
                23923681.729276657,
                791311658718.4193,
                1.773374462991839e-05,
                0.14642752940923615,
                1600.5596278736678,
                29949429.82503553,
                1112815989293.9326,
                1.5303115839590797e-05,
                0.14194150045081785,
                1742.873058605698,
                36634451.931305364,
                1529085389160.7544,
                1.3148448731163076e-05,
                0.13699368732998807,
                1889.5284359054356,
                44614279.74469635,
                2091762812969.9607,
                1.1739194407590062e-05,
                0.13739553134643406,
                2128.794599579694,
                56462810.11822766,
                2973783283306.8145,
                1.0293367506254706e-05,
                0.13533033372723272,
                2355.372854690074,
                70176508.28667311,
                4151852759764.441,
                9.678312586863569e-06,
                0.14293577249119244,
                2794.531827932675,
                93528671.31952812,
                6215821967224.52,
                -1.174086323572049e-05,
                0.1429501325944908,
                3139.4804810720925,
                118031680.16618933,
                -6466892421886.174,
                -2.1188265307407812e-05,
                0.1477108290912869,
                3644.1133424610953,
                153900132.62392554,
                -4828013117542.036,
                -8.614483025123122e-05,
                0.16037100755883044,
                4444.386620899393,
                210846007.89660168,
                -1766340937974.433,
                4.981445776141726e-05,
                0.16053420251962536,
                4997.558254401547,
                266327328.4755411,
                3862250287024.725,
                1.8500019169456637e-05,
                0.15448417164977674,
                5402.289867444643,
                323399508.1475582,
                12152445411933.408,
                -5.647882376069748e-05,
                0.1406372975946189,
                5524.633133597753,
                371512945.9909363,
                -4162951345292.1514,
                2.8048523486337994e-05,
                0.13183417571186926,
                5817.462495763679,
                439447252.3728975,
                9294740538175.03,
            ]
        ).reshape(89, 5)
        b = np.ones(89, dtype=np.float64)
        sol, rnorm = func(A, b)
        assert_allclose(sol, np.array([0.61124315, 8.22262829, 0.0, 0.0, 0.0]))
        assert_allclose(rnorm, 1.0556460808977297)

    @pytest.mark.skipif(not HAS_SCIPY, reason="Scipy not installed")
    def test_nnls_112_114_gh20302(self):
        self._test_nnls_gh20302(nnls_112_114)

    @pytest.mark.skipif(not HAS_SCIPY or __version__ < "1.12", reason=REASON)
    def test_nnls_njit_gh20302(self):
        self._test_nnls_gh20302(njit(lambda a, b: nnls(a, b)))
