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
        n_iter=1000,
        a=0.02,
        c=0.1,
        gamma=0.101,
        grad_clip=20,
        step_clip=2,
        n_grad_avg=3
    ),
    randomSearch_params=dict(
        N_0=1000,
        stages=((25, 25), (10, 50)),
    ),
    fem_verbose=False,
    mesh_verbose=False,
    ll_verbose=False,
    ll_verbose_freq=100,
    cities={},
    discrete_bass_history_mode="generative",
    smith_song_history_mode="generative",
    ml_xgb_params=dict(
        n_estimators=800,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=0,
        n_jobs=-1,
    ),   
    ml_history_mode="generative",
    ml_neighbor_top_k=50,
    ml_neighbor_theta=0.05,
    ml_lag_steps=3)




"""
Shortcuts for models
"""
'''GSB_mle_model_params=dict( # SSB_V1_SPEC
    r0=("pos", 0.01, 10),
    r1=("pos", 0.01, 100),
    r2=("pos", 0.1, 10),
    p=("pos", 1e-5, 1),
    q_I=("pos", 1e-3, 10),
    gamma_J=("pos", 1e-3, 10),
    k_J=("nonneg", 0, 1),
    D=("pos", 1, 1e4),
    S0=("const", 0, 0),
    phi=("pos", 1, 1e4))'''

'''GSB_mle_model_params=dict( # SSB_V2_SPEC
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
    phi=("const", 100, 100))'''

GSB_mle_model_params=dict( # SSB_V3_SPEC
    r0=("pos", 0.01, 10),
    r1=("pos", 0.01, 100),
    r2=("pos", 0.1, 10),
    p=("pos", 1e-5, 0.1),
    q_I=("pos", 1e-3, 10),
    a_I=("pos", 0.1, 10),
    b_I=("pos", 0.1, 10),
    c_I=("pos", 0.01, 10),
    gamma_J=("pos", 1e-3, 10),
    k_J=("pos", 1e-6, 0.01),
    D=("pos", 1, 1e4),
    S0=("const", 0, 0),
    phi=("const", 100, 100))

SmithSong_mle_model_params=dict(
    r0=("pos", 0.01, 10),
    r1=("pos", 0.01, 100),
    r2=("pos", 0.1, 10),
    theta=("nonneg", 0, 5),
    lambda_mix=("pos", 1e-5, 1),
    a_time=("nonneg", 0, 2),
    phi=("const", 100, 100))

DiscreteBass_mle_model_params=dict(
    p=("pos", 1e-5, 0.1),
    q=("pos", 1e-5, 10),
    theta=("nonneg", 0, 10),
    r0=("pos", 0.01, 10),
    r1=("pos", 0.01, 100),
    r2=("pos", 0.1, 10),
    phi=("const", 100, 100))

objective_type='rmse'




"""
Shortcuts for regions
"""

Chicago_mesh_params=dict(
    state_list=['IL', 'WI'],
    county_list=['Cook', 'DuPage', 'Kane', 'Lake', 'McHenry', 'Will', 'DeKalb', 'Grundy', 'Kendall', 'Kenosha'],
    h_km=4,
    simplify_km=12)

Chicago_time_params_sample=dict(
    start_year=2015,
    T_years=9,
    t_min_year=2015,
    t_max_year=2023)

Chicago_time_params_forecast=dict(
    start_year=2015,
    T_years=7,
    t_min_year=2022,
    t_max_year=2023)

Chicago_cities={"Chicago": [-87.629789, 41.878114]}

# ---

Phoenix_mesh_params=dict(
    state_list=['AZ'],
    county_list=['Maricopa', 'Pinal'],
    h_km=7,
    simplify_km=21)

Phoenix_time_params_sample=dict(
    start_year=2005,
    T_years=18,
    t_min_year=2005,
    t_max_year=2022)

Phoenix_time_params_forecast=dict(
    start_year=2005,
    T_years=16,
    t_min_year=2021,
    t_max_year=2022)

Phoenix_cities={"Phoenix": [-112.074036, 33.448376]}

# ---

NewYork_mesh_params=dict(
    state_list=['NY', 'NJ'],
    county_list=['Kings', 'New York', 'Queens', 'Bronx', 'Richmond', 'Nassau', 'Suffolk', 'Westchester', 'Rockland', 'Putnam', 'Orange', 'Dutchess', 'Sullivan', 'Ulster', 'Bergen', 'Essex', 
                 'Hudson', 'Middlesex', 'Morris', 'Passaic', 'Somerset', 'Union', 'Hunterdon', 'Monmouth', 'Ocean', 'Sussex', 'Warren', 'Mercer'],
    h_km=6,
    simplify_km=18)

NewYork_time_params_sample=dict(
    start_year=2002,
    T_years=20,
    t_min_year=2002,
    t_max_year=2021)

NewYork_time_params_forecast=dict(
    start_year=2002,
    T_years=20,
    t_min_year=2022,
    t_max_year=2023)

NewYork_cities={"New York City": [-74.0060, 40.7128]}

# ---

Minneapolis_mesh_params=dict(
    state_list=['MN'],
    county_list=['Anoka', 'Carver', 'Dakota', 'Hennepin', 'Ramsey', 'Scott', 'Washington'],
    h_km=3,
    simplify_km=9)

Minneapolis_time_params_sample=dict(
    start_year=2007,
    T_years=15,
    t_min_year=2007,
    t_max_year=2021)

Minneapolis_time_params_forecast=dict(
    start_year=2007,
    T_years=15,
    t_min_year=2022,
    t_max_year=2023)

Minneapolis_cities={"Minneapolis": [-93.264358, 44.977479]}

# ---

Austin_mesh_params=dict(
    state_list=['TX'],
    county_list=['Travis','Williamson'],
    h_km=3,
    simplify_km=9)

Austin_time_params_sample=dict(
    start_year=2004,
    T_years=18,
    t_min_year=2004,
    t_max_year=2021)

Austin_time_params_forecast=dict(
    start_year=2004,
    T_years=18,
    t_min_year=2022,
    t_max_year=2023)

Austin_cities={"Austin": [-97.7431, 30.2672]}

# ---

LosAngeles_mesh_params=dict(
    state_list=['CA'],
    county_list=['Orange','Los Angeles','Riverside','San Bernardino','Ventura'],
    h_km=10,
    simplify_km=30)

LosAngeles_time_params_sample=dict(
    start_year=2001,
    T_years=21,
    t_min_year=2001,
    t_max_year=2021)

LosAngeles_time_params_forecast=dict(
    start_year=2001,
    T_years=21,
    t_min_year=2022,
    t_max_year=2023)

LosAngeles_cities={"Los Angeles": [-118.2426, 34.0549]}

# ---

SanFrancisco_mesh_params=dict(
    state_list=['CA'],
    county_list=['Alameda', 'Contra Costa', 'Marin', 'San Francisco', 'San Mateo'],
    h_km=2,
    simplify_km=6)

SanFrancisco_time_params_sample=dict(
    start_year=2001,
    T_years=21,
    t_min_year=2001,
    t_max_year=2021)

SanFrancisco_time_params_forecast=dict(
    start_year=2001,
    T_years=21,
    t_min_year=2022,
    t_max_year=2023)

SanFrancisco_cities={"San Francisco": [-122.4194, 37.7749], "San Jose": [121.8863, 37.3382]}

# ---

SanDiego_mesh_params=dict(
    state_list=['CA'],
    county_list=['San Diego'],
    h_km=4,
    simplify_km=12)

SanDiego_time_params_sample=dict(
    start_year=2001,
    T_years=21,
    t_min_year=2001,
    t_max_year=2021)

SanDiego_time_params_forecast=dict(
    start_year=2001,
    T_years=21,
    t_min_year=2022,
    t_max_year=2023)

SanDiego_cities={"San Diego": [-117.1611, 32.7157]}

# ---

Denver_mesh_params=dict(
    state_list=['CO'],
    county_list=['Denver', 'Adams', 'Arapahoe', 'Broomfield', 'Clear Creek', 'Douglas', 'Jefferson'],
    h_km=4,
    simplify_km=12)

Denver_time_params_sample=dict(
    start_year=2007,
    T_years=15,
    t_min_year=2007,
    t_max_year=2021)

Denver_time_params_forecast=dict(
    start_year=2007,
    T_years=15,
    t_min_year=2022,
    t_max_year=2023)

Denver_cities={"Denver": [-104.9915, 39.7420]}

# ---

Orlando_mesh_params=dict(
    state_list=['FL'],
    county_list=['Orange','Osceola'],
    h_km=3,
    simplify_km=9)

Orlando_time_params_sample=dict(
    start_year=2015,
    T_years=7,
    t_min_year=2015,
    t_max_year=2021)

Orlando_time_params_forecast=dict(
    start_year=2015,
    T_years=7,
    t_min_year=2022,
    t_max_year=2023)

Orlando_cities={"Orlando": [-81.3789, 28.5384]}

# ---

SanAntonio_mesh_params=dict(
    state_list=['TX'],
    county_list=['Bexar','Comal','Medina'],
    h_km=3,
    simplify_km=9)

SanAntonio_time_params_sample=dict(
    start_year=2010,
    T_years=12,
    t_min_year=2010,
    t_max_year=2021)

SanAntonio_time_params_forecast=dict(
    start_year=2010,
    T_years=12,
    t_min_year=2022,
    t_max_year=2023)

SanAntonio_cities={"San Antonio": [-98.4911, 29.4243]}




"""
Builder of 4 in-sample and 4 forecasting models. See the dictionaries/values above and replace 'Area' everywhere with 'LosAngeles', for example, to use.
"""
def build_8_models(mesh_params, time_params_sample, time_params_forecast, cities, objective_type=objective_type):
    
    dict_base = dict(mesh_params=mesh_params, cities=cities)
    
    GSB_sample = dict_base | dict(time_params=time_params_sample, objective_type=objective_type, mle_model_params=GSB_mle_model_params)
    GSB_forecast = dict_base | dict(time_params=time_params_forecast, objective_type=objective_type, mle_model_params=GSB_mle_model_params)
    
    SmithSong_sample = GSB_sample | dict(benchmark_model='smith_song', mle_model_params=SmithSong_mle_model_params)
    SmithSong_forecast = GSB_forecast | dict(benchmark_model='smith_song', mle_model_params=SmithSong_mle_model_params)
    
    DiscreteBass_sample = GSB_sample | dict(benchmark_model='discrete_bass', mle_model_params=DiscreteBass_mle_model_params)
    DiscreteBass_forecast = GSB_forecast | dict(benchmark_model='discrete_bass', mle_model_params=DiscreteBass_mle_model_params)

    ml_sample = dict_base | dict(benchmark_model='xgboost', 
                                     ml_train_start_year=time_params_sample['start_year'], 
                                     ml_train_end_year=time_params_sample['start_year']+time_params_sample['T_years']-1,
                                     ml_test_start_year=time_params_sample['t_min_year'],
                                     ml_test_end_year=time_params_sample['t_max_year'],
                                     time_params=time_params_sample)
    ml_forecast = dict_base | dict(benchmark_model='xgboost', 
                                     ml_train_start_year=time_params_forecast['start_year'], 
                                     ml_train_end_year=time_params_forecast['start_year']+time_params_forecast['T_years']-1,
                                     ml_test_start_year=time_params_forecast['t_min_year'],
                                     ml_test_end_year=time_params_forecast['t_max_year'],
                                     time_params=time_params_forecast)
    
    return [GSB_sample, GSB_forecast, SmithSong_sample, SmithSong_forecast, DiscreteBass_sample, DiscreteBass_forecast, ml_sample, ml_forecast]
    



"""
Custom configurations start here
"""

(Chicago_GSB_sample, Chicago_GSB_forecast, 
 Chicago_SmithSong_sample, Chicago_SmithSong_forecast, 
 Chicago_DiscreteBass_sample, Chicago_DiscreteBass_forecast, 
 Chicago_ml_sample, Chicago_ml_forecast) = build_8_models(Chicago_mesh_params, Chicago_time_params_sample, Chicago_time_params_forecast, Chicago_cities)

(Phoenix_GSB_sample, Phoenix_GSB_forecast, 
 Phoenix_SmithSong_sample, Phoenix_SmithSong_forecast, 
 Phoenix_DiscreteBass_sample, Phoenix_DiscreteBass_forecast, 
 Phoenix_ml_sample, Phoenix_ml_forecast) = build_8_models(Phoenix_mesh_params, Phoenix_time_params_sample, Phoenix_time_params_forecast, Phoenix_cities)

(NewYork_GSB_sample, NewYork_GSB_forecast, 
 NewYork_SmithSong_sample, NewYork_SmithSong_forecast, 
 NewYork_DiscreteBass_sample, NewYork_DiscreteBass_forecast, 
 NewYork_ml_sample, NewYork_ml_forecast) = build_8_models(NewYork_mesh_params, NewYork_time_params_sample, NewYork_time_params_forecast, NewYork_cities)

(Minneapolis_GSB_sample, Minneapolis_GSB_forecast, 
 Minneapolis_SmithSong_sample, Minneapolis_SmithSong_forecast, 
 Minneapolis_DiscreteBass_sample, Minneapolis_DiscreteBass_forecast, 
 Minneapolis_ml_sample, Minneapolis_ml_forecast) = build_8_models(Minneapolis_mesh_params, Minneapolis_time_params_sample, Minneapolis_time_params_forecast, Minneapolis_cities)

(Austin_GSB_sample, Austin_GSB_forecast, 
 Austin_SmithSong_sample, Austin_SmithSong_forecast, 
 Austin_DiscreteBass_sample, Austin_DiscreteBass_forecast, 
 Austin_ml_sample, Austin_ml_forecast) = build_8_models(Austin_mesh_params, Austin_time_params_sample, Austin_time_params_forecast, Austin_cities)

(LosAngeles_GSB_sample, LosAngeles_GSB_forecast, 
 LosAngeles_SmithSong_sample, LosAngeles_SmithSong_forecast, 
 LosAngeles_DiscreteBass_sample, LosAngeles_DiscreteBass_forecast, 
 LosAngeles_ml_sample, LosAngeles_ml_forecast) = build_8_models(LosAngeles_mesh_params, LosAngeles_time_params_sample, LosAngeles_time_params_forecast, LosAngeles_cities)

(SanFrancisco_GSB_sample, SanFrancisco_GSB_forecast, 
 SanFrancisco_SmithSong_sample, SanFrancisco_SmithSong_forecast, 
 SanFrancisco_DiscreteBass_sample, SanFrancisco_DiscreteBass_forecast, 
 SanFrancisco_ml_sample, SanFrancisco_ml_forecast) = build_8_models(SanFrancisco_mesh_params, SanFrancisco_time_params_sample, SanFrancisco_time_params_forecast, SanFrancisco_cities)

(SanDiego_GSB_sample, SanDiego_GSB_forecast, 
 SanDiego_SmithSong_sample, SanDiego_SmithSong_forecast, 
 SanDiego_DiscreteBass_sample, SanDiego_DiscreteBass_forecast, 
 SanDiego_ml_sample, SanDiego_ml_forecast) = build_8_models(SanDiego_mesh_params, SanDiego_time_params_sample, SanDiego_time_params_forecast, SanDiego_cities)

(Denver_GSB_sample, Denver_GSB_forecast, 
 Denver_SmithSong_sample, Denver_SmithSong_forecast, 
 Denver_DiscreteBass_sample, Denver_DiscreteBass_forecast, 
 Denver_ml_sample, Denver_ml_forecast) = build_8_models(Denver_mesh_params, Denver_time_params_sample, Denver_time_params_forecast, Denver_cities)

(Orlando_GSB_sample, Orlando_GSB_forecast, 
 Orlando_SmithSong_sample, Orlando_SmithSong_forecast, 
 Orlando_DiscreteBass_sample, Orlando_DiscreteBass_forecast, 
 Orlando_ml_sample, Orlando_ml_forecast) = build_8_models(Orlando_mesh_params, Orlando_time_params_sample, Orlando_time_params_forecast, Orlando_cities)

(SanAntonio_GSB_sample, SanAntonio_GSB_forecast, 
 SanAntonio_SmithSong_sample, SanAntonio_SmithSong_forecast, 
 SanAntonio_DiscreteBass_sample, SanAntonio_DiscreteBass_forecast, 
 SanAntonio_ml_sample, SanAntonio_ml_forecast) = build_8_models(SanAntonio_mesh_params, SanAntonio_time_params_sample, SanAntonio_time_params_forecast, SanAntonio_cities)