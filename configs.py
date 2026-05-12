#!/usr/bin/env python3
"""
configs.py

Configurations for running the procedure.

CAUTION: make sure to avoid mistakes in naming keys of dictionaries (mle_model_params, fem_model_params, time_params and spsa_params).
Many functions will accept dictionaries with missing/unused keys and substitute default ones instead.
For example, naming parameter 'r_0' instead of 'r0' may have unexpected consequences and mismatch between MLE outputs and FEM diagnostic results.
"""


"""
Default configuration
"""

default = dict(
    mesh_params=dict(
        state_list=['CA'],
        h_km=12,
        simplify_km=36, 
        epsg_project=5070,
    ),
    mle_model_params=dict(
        r0=("pos", 0.01, 10),
        r1=("pos", 0.01, 100),
        r2=("pos", 0.1, 10),
        p=("pos", 1e-5, 1),
        q_I=("pos", 1e-3, 10),
        gamma_J=("pos", 1e-3, 10),
        k_J=("nonneg", 0, 1),
        D=("pos", 1, 1e4),
        S0=("const", 0, 0),
        phi=("pos", 1, 1e4),
    ),
    fem_model_params=dict(
        r0=1,
        r1=0,
        p=0.03,
        q_I=0.5,
        gamma_J=1,
        k_J=0,
        D=1,
        S0=0,
    ),
    time_params=dict(
        start_year=1998,
        tau=0.05,
        T_years=26,
        picard_max_iter=20,
        picard_tol=1e-8,
        t_min_year=1998,
        t_max_year=2024,
    ),
    spsa_params=dict(
        n_iter=2000,
        a=0.05,
        c=0.05,
        gamma=0,
        grad_clip=200,
        step_clip=100,
    ),
    randomSearch_params=dict(
        N_0=2000,
        stages=((40, 50), (5, 400)),
    ),
    fem_verbose=False,
    mesh_verbose=False,
    ll_verbose=False,
    ll_verbose_freq=100,
    cities={},
)


"""
Custom configurations start here
"""

CA = dict(
    mesh_params=dict(
        state_list=['CA'],
        h_km=12,
        simplify_km=36, 
    ),
    fem_model_params=dict( # TO EDIT
        r0=0.21682372494611601,
        r1=0,
        p=1.7966300331458806e-05,
        q_I=0.11642553092295363,
        gamma_J=0.00012305488346120747,
        k_J=8.576616014263596e-05,
        D=0.03381577568515316,
        S0=12.82367787707364,
    ),
    time_params=dict(
        start_year=2003.375,
        T_years=21.625,
        t_min_year=1998,
        t_max_year=2024,
    ),
    cities={
        "Los Angeles": [-118.2426, 34.0549],
        "San Francisco": [-122.4194, 37.7749],
        "San Diego": [-117.1611, 32.7157],
        "San Jose": [-121.8853, 37.3387],
        "Fresno": [-119.7871, 36.7378],
        "Sacramento": [-121.4944, 38.5781],
    },
)


CA_v2 = dict(
    mesh_params=dict(
        state_list=['CA'],
        h_km=12,
        simplify_km=36, 
    ),
    mle_model_params=dict(
        r0=("pos", 0.01, 10),
        r1=("pos", 0.01, 100),
        r2=("pos", 0.1, 10),
        p=("pos", 1e-5, 1),
        q_I=("pos", 1e-3, 10),
        a_I=("nonneg", 0, 10),
        b_I=("nonneg", 0, 10),
        c_I=("pos", 0.01, 10),
        gamma_J1=("pos", 1e-3, 10),
        gamma_J2=("pos", 1e-3, 10),
        k_J=("nonneg", 0, 1),
        D1=("pos", 1, 1e4),
        D2=("pos", 1, 1e4),
        S0=("const", 0, 0),
        phi=("pos", 1, 1e4)),
    fem_model_params=dict(
        r0=0.21682372494611601,
        r1=0,
        p=1.7966300331458806e-05,
        q_I=0.11642553092295363,
        gamma_J=0.00012305488346120747,
        k_J=8.576616014263596e-05,
        D=0.03381577568515316,
        S0=12.82367787707364,
    ),
    time_params=dict(
        start_year=2003.375,
        T_years=21.625,
        t_min_year=2003,
        t_max_year=2024,
    ),
    cities={
        "Los Angeles": [-118.2426, 34.0549],
        "San Francisco": [-122.4194, 37.7749],
        "San Diego": [-117.1611, 32.7157],
        "San Jose": [-121.8853, 37.3387],
        "Fresno": [-119.7871, 36.7378],
        "Sacramento": [-121.4944, 38.5781],
    },
)


IL = dict(
    mesh_params=dict(
        state_list=['IL'],
        h_km=7,
        simplify_km=21, 
    ),
    fem_model_params=dict( # TO EDIT
        r0=0.21682372494611601,
        r1=0,
        p=1.7966300331458806e-05,
        q_I=0.11642553092295363,
        gamma_J=0.00012305488346120747,
        k_J=8.576616014263596e-05,
        D=0.03381577568515316,
        S0=12.82367787707364,
    ),
    time_params=dict(
        start_year=2012.700,
        T_years=11.300,
        t_min_year=2006,
        t_max_year=2024,
    ),
    cities={"Chicago": [-87.6324, 41.8832]},
)


IL_v2 = dict(
    mesh_params=dict(
        state_list=['IL'],
        h_km=7,
        simplify_km=21, 
    ),
    mle_model_params=dict(
        r0=("pos", 0.01, 10),
        r1=("pos", 0.01, 100),
        r2=("pos", 0.1, 10),
        p=("pos", 1e-5, 1),
        q_I=("pos", 1e-3, 10),
        a_I=("nonneg", 0, 10),
        b_I=("nonneg", 0, 10),
        c_I=("pos", 0.01, 10),
        gamma_J1=("pos", 1e-3, 10),
        gamma_J2=("pos", 1e-3, 10),
        k_J=("nonneg", 0, 1),
        D1=("pos", 1, 1e4),
        D2=("pos", 1, 1e4),
        S0=("const", 0, 0),
        phi=("pos", 1, 1e4)),
    time_params=dict(
        start_year=2006,
        T_years=18,
        t_min_year=2006,
        t_max_year=2024,
    ),
    cities={"Chicago": [-87.6324, 41.8832]},
)


NY = dict(
    mesh_params=dict(
        state_list=['NY'],
        h_km=6,
        simplify_km=18, 
    ),
    fem_model_params=dict( # TO EDIT
        r0=0.21682372494611601,
        r1=0,
        p=1.7966300331458806e-05,
        q_I=0.11642553092295363,
        gamma_J=0.00012305488346120747,
        k_J=8.576616014263596e-05,
        D=0.03381577568515316,
        S0=12.82367787707364,
    ),
    time_params=dict(
        start_year=2004,#2008,
        T_years=20,#16,
        t_min_year=2002,
        t_max_year=2024,
    ),
    cities={
        "New York": [-74.0060, 40.7128],
        "Buffalo": [-78.8789, 42.8869],
        "Rochester": [-77.6088, 43.1566],
        "Albany": [-73.7545, 42.6518],
        "Kiryas Joel": [-74.1679, 41.3420],
        "Syracuse": [-76.1474, 43.0495] 
    },
)


NY_v2 = dict(
    mesh_params=dict(
        state_list=['NY'],
        h_km=6,
        simplify_km=18, 
    ),
    mle_model_params=dict(
        r0=("pos", 0.01, 10),
        r1=("pos", 0.01, 100),
        r2=("pos", 0.1, 10),
        p=("pos", 1e-5, 1),
        q_I=("pos", 1e-3, 10),
        a_I=("nonneg", 0, 10),
        b_I=("nonneg", 0, 10),
        c_I=("pos", 0.01, 10),
        gamma_J1=("pos", 1e-3, 10),
        gamma_J2=("pos", 1e-3, 10),
        k_J=("nonneg", 0, 1),
        D1=("pos", 1, 1e4),
        D2=("pos", 1, 1e4),
        S0=("const", 0, 0),
        phi=("pos", 1, 1e4)),
    time_params=dict(
        start_year=2004,
        T_years=20,
        t_min_year=2004,
        t_max_year=2024,
    ),
    cities={
        "New York": [-74.0060, 40.7128],
        "Buffalo": [-78.8789, 42.8869],
        "Rochester": [-77.6088, 43.1566],
        "Albany": [-73.7545, 42.6518],
        "Kiryas Joel": [-74.1679, 41.3420],
        "Syracuse": [-76.1474, 43.0495] 
    },
)


FL = dict(
    mesh_params=dict(
        state_list=['FL'],
        h_km=8,
        simplify_km=24, 
    ),
    fem_model_params=dict( # TO EDIT
        r0=0.21682372494611601,
        r1=0,
        p=1.7966300331458806e-05,
        q_I=0.11642553092295363,
        gamma_J=0.00012305488346120747,
        k_J=8.576616014263596e-05,
        D=0.03381577568515316,
        S0=12.82367787707364,
    ),
    time_params=dict(
        start_year=2014,#2006,
        T_years=10,#18,
        t_min_year=2002,
        t_max_year=2024,
    ),
    cities={
        "Miami": [-80.1918, 25.7617],
        "Tampa": [-82.4588, 27.9517],
        "Orlando": [-81.3789, 28.5384],
        "Jacksonville": [-81.6592, 30.3298]
    },
)


FL_v2 = dict(
    mesh_params=dict(
        state_list=['FL'],
        h_km=8,
        simplify_km=24, 
    ),
    mle_model_params=dict(
        r0=("pos", 0.01, 10),
        r1=("pos", 0.01, 100),
        r2=("pos", 0.1, 10),
        p=("pos", 1e-5, 1),
        q_I=("pos", 1e-3, 10),
        a_I=("nonneg", 0, 10),
        b_I=("nonneg", 0, 10),
        c_I=("pos", 0.01, 10),
        gamma_J1=("pos", 1e-3, 10),
        gamma_J2=("pos", 1e-3, 10),
        k_J=("nonneg", 0, 1),
        D1=("pos", 1, 1e4),
        D2=("pos", 1, 1e4),
        S0=("const", 0, 0),
        phi=("pos", 1, 1e4)),
    time_params=dict(
        start_year=2012,
        T_years=12,
        t_min_year=2012,
        t_max_year=2024,
    ),
    cities={
        "Miami": [-80.1918, 25.7617],
        "Tampa": [-82.4588, 27.9517],
        "Orlando": [-81.3789, 28.5384],
        "Jacksonville": [-81.6592, 30.3298]
    },
)


MN = dict(
    mesh_params=dict(
        state_list=['MN'],
        h_km=8,
        simplify_km=24, 
    ),
    fem_model_params=dict( # TO EDIT
        r0=0.21682372494611601,
        r1=0,
        p=1.7966300331458806e-05,
        q_I=0.11642553092295363,
        gamma_J=0.00012305488346120747,
        k_J=8.576616014263596e-05,
        D=0.03381577568515316,
        S0=12.82367787707364,
    ),
    time_params=dict(
        start_year=2003,
        T_years=20,
        t_min_year=2002,
        t_max_year=2023,
    ),
    cities={
        "Minneapolis": [-93.2650, 44.9778],
        "Duluth": [-92.1055, 46.7845],
        "Fargo": [-97.1216, 47.0712],
        "Rochester": [-92.4588, 44.0193],
        "St Cloud": [-94.1632, 45.5579]
    },
)


MN_v2 = dict(
    mesh_params=dict(
        state_list=['MN'],
        h_km=8,
        simplify_km=24, 
    ),
    mle_model_params=dict(
        r0=("pos", 0.01, 10),
        r1=("pos", 0.01, 100),
        r2=("pos", 0.1, 10),
        p=("pos", 1e-5, 1),
        q_I=("pos", 1e-3, 10),
        a_I=("nonneg", 0, 10),
        b_I=("nonneg", 0, 10),
        c_I=("pos", 0.01, 10),
        gamma_J1=("pos", 1e-3, 10),
        gamma_J2=("pos", 1e-3, 10),
        k_J=("nonneg", 0, 1),
        D1=("pos", 1, 1e4),
        D2=("pos", 1, 1e4),
        S0=("const", 0, 0),
        phi=("pos", 1, 1e4)),
    time_params=dict(
        start_year=2003,
        T_years=20,
        t_min_year=2003,
        t_max_year=2023,
    ),
    cities={
        "Minneapolis": [-93.2650, 44.9778],
        "Duluth": [-92.1055, 46.7845],
        "Fargo": [-97.1216, 47.0712],
        "Rochester": [-92.4588, 44.0193],
        "St Cloud": [-94.1632, 45.5579]
    },
)


TX_v2 = dict(
    mesh_params=dict(
        state_list=['TX'],
        h_km=15,
        simplify_km=45, 
    ),
    mle_model_params=dict(
        r0=("pos", 0.01, 10),
        r1=("pos", 0.01, 100),
        r2=("pos", 0.1, 10),
        p=("pos", 1e-5, 1),
        q_I=("pos", 1e-3, 10),
        a_I=("nonneg", 0, 10),
        b_I=("nonneg", 0, 10),
        c_I=("pos", 0.01, 10),
        gamma_J1=("pos", 1e-3, 10),
        gamma_J2=("pos", 1e-3, 10),
        k_J=("nonneg", 0, 1),
        D1=("pos", 1, 1e4),
        D2=("pos", 1, 1e4),
        S0=("const", 0, 0),
        phi=("pos", 1, 1e4)),
    time_params=dict(
        start_year=2008,
        T_years=15,
        t_min_year=2008,
        t_max_year=2023,
    ),
    cities={
        "Dallas": [-96.8089, 32.7792],
        "Houston": [-95.3701, 29.7601],
        "San Antonio": [-98.4946, 29.4252],
        "Austin": [-97.7431, 30.2672],
        "McAllen": [-98.2300, 26.2034],
        "El Paso": [-106.4850, 31.7619],
        "Killeen": [-97.6982, 31.1242],
        "Corpus Christi": [-97.4030, 27.7964]
    },
)


AZ_v2 = dict(
    mesh_params=dict(
        state_list=['AZ'],
        h_km=9,
        simplify_km=27, 
    ),
    mle_model_params=dict(
        r0=("pos", 0.01, 10),
        r1=("pos", 0.01, 100),
        r2=("pos", 0.1, 10),
        p=("pos", 1e-5, 1),
        q_I=("pos", 1e-3, 10),
        a_I=("nonneg", 0, 10),
        b_I=("nonneg", 0, 10),
        c_I=("pos", 0.01, 10),
        gamma_J1=("pos", 1e-3, 10),
        gamma_J2=("pos", 1e-3, 10),
        k_J=("nonneg", 0, 1),
        D1=("pos", 1, 1e4),
        D2=("pos", 1, 1e4),
        S0=("const", 0, 0),
        phi=("pos", 1, 1e4)),
    time_params=dict(
        start_year=2008,
        T_years=15,
        t_min_year=2008,
        t_max_year=2023,
    ),
    cities={
        "Phoenix": [-112.0740, 33.4484],
        "Tucson": [-110.9742, 32.2540],
        "Prescott": [-112.4682, 34.5394],
        "Lake Havasu City": [-114.3192, 34.4770],
        "Kingman": [-114.0523, 35.1913],
        "Yuma": [-114.6277, 32.6927],
        "Flagstaff": [-111.6513, 35.1983]
    },
)

VA_v2 = dict(
    mesh_params=dict(
        state_list=['VA'],
        h_km=6,
        simplify_km=18, 
    ),
    mle_model_params=dict(
        r0=("pos", 0.01, 10),
        r1=("pos", 0.01, 100),
        r2=("pos", 0.1, 10),
        p=("pos", 1e-5, 1),
        q_I=("pos", 1e-3, 10),
        a_I=("nonneg", 0, 10),
        b_I=("nonneg", 0, 10),
        c_I=("pos", 0.01, 10),
        gamma_J1=("pos", 1e-3, 10),
        gamma_J2=("pos", 1e-3, 10),
        k_J=("nonneg", 0, 1),
        D1=("pos", 1, 1e4),
        D2=("pos", 1, 1e4),
        S0=("const", 0, 0),
        phi=("pos", 1, 1e4)),
    time_params=dict(
        start_year=2008,
        T_years=11,
        t_min_year=2008,
        t_max_year=2021,
    ),
    cities={
        "Arlington": [-77.0831, 38.8827],
        "Virginia Beach": [-75.9792, 36.8516],
        "Richmond": [-77.4360, 37.5407],
        "Roanoke": [-79.9449, 37.2709],
        "Lynchburg": [-79.1422, 37.4149],
        "Charlottesville": [-78.4769, 38.0302]
    },
)

OH_v2 = dict(
    mesh_params=dict(
        state_list=['OH'],
        h_km=6,
        simplify_km=18, 
    ),
    mle_model_params=dict(
        r0=("pos", 0.01, 10),
        r1=("pos", 0.01, 100),
        r2=("pos", 0.1, 10),
        p=("pos", 1e-5, 1),
        q_I=("pos", 1e-3, 10),
        a_I=("nonneg", 0, 10),
        b_I=("nonneg", 0, 10),
        c_I=("pos", 0.01, 10),
        gamma_J1=("pos", 1e-3, 10),
        gamma_J2=("pos", 1e-3, 10),
        k_J=("nonneg", 0, 1),
        D1=("pos", 1, 1e4),
        D2=("pos", 1, 1e4),
        S0=("const", 0, 0),
        phi=("pos", 1, 1e4)),
    time_params=dict(
        start_year=2008,
        T_years=16,
        t_min_year=2008,
        t_max_year=2024,
    ),
    cities={
        "Cleveland": [-81.6944, 41.4993],
        "Columbus": [-83.0032, 39.9625],
        "Cincinnati": [-84.5120, 39.1031],
        "Dayton" : [-84.1936, 39.7592]
    },
)

OH_smith_song = dict(
    mesh_params=dict(
        state_list=["OH"],
        h_km=6,
        simplify_km=18,
    ),
    mle_model_params=dict(
        r0=("pos", 0.01, 10),
        r1=("pos", 0.01, 100),
        r2=("pos", 0.1, 10),
        theta=("nonneg", 0, 5),          # distance decay
        lambda_mix=("pos", 1e-5, 1), # innovation probability
        a_time=("nonneg", 0, 2),         # temporal growth
        phi=("pos", 1, 1e4),
    ),
    time_params=dict(
        start_year=2008,
        T_years=16,
        t_min_year=2008, #2002
        t_max_year=2024,
    ),
    cities={
        "Cleveland": [-81.6944, 41.4993],
        "Columbus": [-83.0032, 39.9625],
        "Cincinnati": [-84.5120, 39.1031],
        "Dayton" : [-84.1936, 39.7592]
    },
    benchmark_model="smith_song",
    smith_song_history_mode="conditional",
)

TX_smith_song = dict(
    mesh_params=dict(
        state_list=['TX'],
        h_km=15,
        simplify_km=45, 
    ),
    mle_model_params=dict(
        r0=("pos", 0.01, 10),
        r1=("pos", 0.01, 100),
        r2=("pos", 0.1, 10),
        theta=("nonneg", 0, 5),          # distance decay
        lambda_mix=("pos", 1e-5, 1), # innovation probability
        a_time=("nonneg", 0, 2),         # temporal growth
        phi=("pos", 1, 1e4),
    ),
    time_params=dict(
        start_year=2008,
        T_years=15,
        t_min_year=2008, #2002
        t_max_year=2023,
    ),
    cities={
        "Dallas": [-96.8089, 32.7792],
        "Houston": [-95.3701, 29.7601],
        "San Antonio": [-98.4946, 29.4252],
        "Austin": [-97.7431, 30.2672],
        "McAllen": [-98.2300, 26.2034],
        "El Paso": [-106.4850, 31.7619],
        "Killeen": [-97.6982, 31.1242],
        "Corpus Christi": [-97.4030, 27.7964]
    },
    benchmark_model="smith_song",
    smith_song_history_mode="conditional",
)

FL_smith_song = dict(
    mesh_params=dict(
        state_list=['FL'],
        h_km=8,
        simplify_km=24, 
    ),
    mle_model_params=dict(
        r0=("pos", 0.01, 10),
        r1=("pos", 0.01, 100),
        r2=("pos", 0.1, 10),
        theta=("nonneg", 0, 5),          # distance decay
        lambda_mix=("pos", 1e-5, 1), # innovation probability
        a_time=("nonneg", 0, 2),         # temporal growth
        phi=("pos", 1, 1e4),
    ),
    time_params=dict(
        start_year=2012,
        T_years=12,
        t_min_year=2012, #2002
        t_max_year=2024,
    ),
    cities={
        "Miami": [-80.1918, 25.7617],
        "Tampa": [-82.4588, 27.9517],
        "Orlando": [-81.3789, 28.5384],
        "Jacksonville": [-81.6592, 30.3298]
    },
    benchmark_model="smith_song",
    smith_song_history_mode="conditional",
)

OH_smith_song_generative = dict(
    mesh_params=dict(
        state_list=["OH"],
        h_km=6,
        simplify_km=18,
    ),
    mle_model_params=dict(
        r0=("pos", 0.01, 10),
        r1=("pos", 0.01, 100),
        r2=("pos", 0.1, 10),
        theta=("nonneg", 0, 5),          # distance decay
        lambda_mix=("pos", 1e-5, 1), # innovation probability
        a_time=("nonneg", 0, 2),         # temporal growth
        phi=("pos", 1, 1e4),
    ),
    time_params=dict(
        start_year=2008,
        T_years=16,
        t_min_year=2008, #2002
        t_max_year=2024,
    ),
    cities={
        "Cleveland": [-81.6944, 41.4993],
        "Columbus": [-83.0032, 39.9625],
        "Cincinnati": [-84.5120, 39.1031],
        "Dayton" : [-84.1936, 39.7592]
    },
    benchmark_model="smith_song",
    smith_song_history_mode="generative",
)