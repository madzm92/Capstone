import geopandas as gpd
from sqlalchemy import create_engine, Column, Integer, String, Float, text
from sqlalchemy.ext.declarative import declarative_base
from geoalchemy2 import WKTElement
from sqlalchemy.orm import sessionmaker
import os
from capstone.database_set_up.table_definitions import Base, ShapeFile
from shapely import wkt


#TODO: generalize
# geoms_file_list = ['2_ACTON_detail', '51_CARLISLE_detail', '49_CAMBRIDGE_detail', '248_REVERE_detail']
rapid_transit_communities = {"folder_name":['40_BRAINTREE_detail', '46_BROOKLINE_detail', '49_CAMBRIDGE_detail', '57_CHELSEA_detail', '93_EVERETT_detail', '165_MALDEN_detail', '176_MEDFORD_detail','189_MILTON_detail','207_NEWTON_detail','243_QUINCY_detail','248_REVERE_detail','274_SOMERVILLE_detail'], 'town_name':['Braintree', 'Brookline', 'Cambridge', 'Chelsea', 'Everett', 'Malden','Medford', 'Milton','Newton', 'Quincy','Revere','Somerville']}
categories = ['Rapid Transit Community', 'Adjacent Community','Commuter Rail Community', 'Adjacent Small Town']
adjacent_community = {
    "folder_name":['7_AMESBURY_detail', '10_ARLINGTON_detail','17_AUBURN_detail','23_BEDFORD_detail','25_BELLINGHAM_detail','48_BURLINGTON_detail','56_CHELMSFORD_detail','71_DANVERS_detail','79_DRACUT_detail','82_DUXBURY_detail','83_EAST_BRIDGEWATER_detail','88_EASTON_detail','105_GEORGETOWN_detail','110_GRAFTON_detail','122_HANOVER_detail','134_HOLDEN_detail','136_HOLLISTON_detail','139_HOPKINTON_detail','142_HULL_detail','155_LEXINGTON_detail','164_LYNNFIELD_detail','168_MARBLEHEAD_detail','170_MARLBOROUGH_detail','171_MARSHFIELD_detail', '174_MAYNARD_detail', '175_MEDFIELD_detail', '177_MEDWAY_detail', '181_METHUEN_detail', '184_MIDDLETON_detail', '186_MILLBURY_detail', '187_MILLIS_detail', '210_NORTH_ANDOVER_detail', '211_NORTH_ATTLEBOROUGH_detail', '213_NORTH_READING_detail', '215_NORTHBOROUGH_detail', '216_NORTHBRIDGE_detail', '218_NORTON_detail', '219_NORWELL_detail', '229_PEABODY_detail', '231_PEMBROKE_detail', '239_PLYMOUTH_detail', '245_RAYNHAM_detail', '251_ROCKLAND_detail', '259_SALISBURY_detail', '262_SAUGUS_detail', '265_SEEKONK_detail', '271_SHREWSBURY_detail', '284_STONEHAM_detail', '288_SUDBURY_detail', '295_TEWKSBURY_detail', '301_TYNGSBOROUGH_detail', '310_WAREHAM_detail', '314_WATERTOWN_detail', '315_WAYLAND_detail', '321_WEST_BOYLSTON_detail', '330_WESTFORD_detail', '346_WINTHROP_detail', '350_WRENTHAM_detail'], 
    'town_name':['Amesbury', 'Arlington','Auburn','Bedford','Bellingham','Burlington','Chelmsford','Danvers','Dracut','Duxbury','East Bridgewater','Easton','Georgetown','Grafton','Hanover','Holden','Holliston','Hopkinton','Hull','Lexington','Lynnfield','Marblehead','Marlborough','Marshfield','Maynard', 'Medfield', 'Medway', 'Methuen', 'Middleton', 'Millbury', 'Millis', 'North Andover', 'North Attleborough', 'North Reading', 'Northborough', 'Northbridge', 'Norton', 'Norwell', 'Peabody', 'Pembroke', 'Plymouth', 'Raynham', 'Rockland', 'Salisbury', 'Saugus', 'Seekonk', 'Shrewsbury', 'Stoneham', 'Sudbury', 'Tewksbury', 'Tyngsborough', 'Wareham', 'Watertown', 'Wayland', 'West Boylston', 'Westford', 'Winthrop', 'Wrentham']}
##TODO: SKIPPED HOLDEN/HANOVER
#ADD CARLISLE BACK IN
adjacent_small_town = {
    "folder_name": ['11_ASHBURNHAM_detail', '12_ASHBY_detail', '27_BERKLEY_detail', '36_BOURNE_detail', '37_BOXBOROUGH_detail', '38_BOXFORD_detail', '52_CARVER_detail', '78_DOVER_detail', '92_ESSEX_detail', '115_GROTON_detail', '116_GROVELAND_detail', '125_HARVARD_detail','146_LAKEVILLE_detail', '147_LANCASTER_detail', '151_LEICESTER_detail', '162_LUNENBURG_detail', '180_MERRIMAC_detail', '196_NAHANT_detail', '205_NEWBURY_detail', '228_PAXTON_detail', '240_PLYMPTON_detail', '241_PRINCETON_detail', '247_REHOBOTH_detail', '250_ROCHESTER_detail', '269_SHERBORN_detail', '282_STERLING_detail', '286_STOW_detail', '290_SUTTON_detail', '298_TOPSFIELD_detail', '299_TOWNSEND_detail', '303_UPTON_detail', '322_WEST_BRIDGEWATER_detail', '324_WEST_NEWBURY_detail', '332_WESTMINSTER_detail'], 
    "town_name": ['Ashburnham', 'Ashby', 'Berkley', 'Bourne', 'Boxborough', 'Boxford', 'Carver', 'Dover', 'Essex', 'Groton', 'Groveland', 'Harvard', 'Lakeville', 'Lancaster', 'Leicester', 'Lunenburg', 'Merrimac', 'Nahant', 'Newbury', 'Paxton', 'Plympton', 'Princeton', 'Rehoboth', 'Rochester', 'Sherborn', 'Sterling', 'Stow', 'Sutton', 'Topsfield', 'Townsend', 'Upton', 'West Bridgewater	', 'West Newbury', 'Westminster'],
}
#ADD ACTON BACK IN
#MISSING FALL RIVER & NEW BEDFORD
commuter_rail = {
    "folder_name": ['1_ABINGTON_detail', '9_ANDOVER_detail', '14_ASHLAND_detail', '16_ATTLEBORO_detail', '19_AYER_detail', '26_BELMONT_detail', '30_BEVERLY_detail', '31_BILLERICA_detail', '42_BRIDGEWATER_detail', '44_BROCKTON_detail', '50_CANTON_detail', '65_COHASSET_detail', '67_CONCORD_detail', '73_DEDHAM_detail', '97_FITCHBURG_detail', '99_FOXBOROUGH_detail', '100_FRAMINGHAM_detail', '101_FRANKLIN_detail', '102_FREETOWN_detail', '107_GLOUCESTER_detail', '118_HALIFAX_detail', '119_HAMILTON_detail', '123_HANSON_detail', '128_HAVERHILL_detail', '131_HINGHAM_detail', '133_HOLBROOK_detail', '144_IPSWICH_detail', '145_KINGSTON_detail', '149_LAWRENCE_detail', '153_LEOMINSTER_detail', '157_LINCOLN_detail', '158_LITTLETON_detail', '160_LOWELL_detail', '163_LYNN_detail', '166_MANCHESTER_detail', '167_MANSFIELD_detail', '178_MELROSE_detail', '182_MIDDLEBOROUGH_detail', '198_NATICK_detail', '199_NEEDHAM_detail', '206_NEWBURYPORT_detail', '208_NORFOLK_detail', '220_NORWOOD_detail', '244_RANDOLPH_detail', '246_READING_detail', '252_ROCKPORT_detail', '254_ROWLEY_detail', '258_SALEM_detail', '264_SCITUATE_detail', '266_SHARON_detail', '270_SHIRLEY_detail', '277_SOUTHBOROUGH_detail', '285_STOUGHTON_detail', '291_SWAMPSCOTT_detail', '293_TAUNTON_detail', '305_WAKEFIELD_detail', '307_WALPOLE_detail', '308_WALTHAM_detail', '317_WELLESLEY_detail', '320_WENHAM_detail', '328_WESTBOROUGH_detail', '333_WESTON_detail', '335_WESTWOOD_detail', '336_WEYMOUTH_detail', '338_WHITMAN_detail', '342_WILMINGTON_detail', '344_WINCHESTER_detail', '347_WOBURN_detail', '348_WORCESTER_detail'],
    "town_name": ['Abington', 'Andover', 'Ashland', 'Attleboro', 'Ayer', 'Belmont', 'Beverly', 'Billerica', 'Bridgewater', 'Brockton', 'Canton', 'Cohasset', 'Concord', 'Dedham', 'Fitchburg', 'Foxborough', 'Framingham', 'Franklin', 'Freetown', 'Gloucester', 'Halifax', 'Hamilton', 'Hanson', 'Haverhill', 'Hingham', 'Holbrook', 'Ipswich', 'Kingston', 'Lawrence', 'Leominster', 'Lincoln', 'Littleton', 'Lowell', 'Lynn', 'Manchester', 'Mansfield', 'Melrose', 'Middleborough', 'Natick', 'Needham', 'Newburyport', 'Norfolk', 'Norwood', 'Randolph', 'Reading', 'Rockport', 'Rowley', 'Salem', 'Scituate', 'Sharon', 'Shirley', 'Southborough', 'Stoughton', 'Swampscott', 'Taunton', 'Wakefield', 'Walpole', 'Waltham', 'Wellesley', 'Wenham', 'Westborough', 'Weston', 'Westwood', 'Weymouth', 'Whitman', 'Wilmington', 'Winchester', 'Woburn', 'Worcester']
}
# Path to the directory containing all the shapefile folders
shapefile_base_dir = 'data_sources/town_shapefiles/'
# file_name = '274_SOMERVILLE_detail.shp'

for folder_name, town_name in zip(commuter_rail['folder_name'], commuter_rail['town_name']): 
    print("folder_name", folder_name)
    print("town_name", town_name)
    breakpoint()
    file_name = folder_name
    full_path = os.path.join(shapefile_base_dir + folder_name, folder_name +'.shp')
    main_shapefile = gpd.read_file(full_path)

    #TODO: generalize
    main_shapefile['town_name'] = town_name

    #add category type
    main_shapefile['category'] = 'Town'
    main_shapefile['community_category'] = 'Commuter Rail Community'

    # create table & insert data
    DB_URI = "postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db"

    # Set up SQLAlchemy and GeoAlchemy2
    engine = create_engine(DB_URI)
    column_replace_dict = {
        'LOC_ID':'loc_id', 
        'Address':'address', 
        'Owner':'owner', 
        'UseCodes':'use_codes', 
        'UseDesc':'use_type',
        'TRANSIT':'transit',
        'ACRES':'acres',
        'SQFT':'sqft', 
        'PublicInst':'public_inst',
        'NonPubExc':'non_public_exec',
        'Tot_Exclud':'total_excluded',
        'Tot_Sensit':'total_sensit',
        'ROW':'row',
        'OpenSpace':'open_space',
        'Hydrology': 'hydrology',
        'Wetlands':'wetlands',
        'TitleV': 'title_v',
        'Wellhead1':'well_head_one',
        'Flood_SHFA':'flood_shfa',
        'Farmland': 'farmland',
        'SurfWatBC': 'surf_wat_bc',
        'Wellhead2':'well_head_two',
        'IntWellhea': 'int_well_hea',
        'Habitat':'habitat'
        }
    main_shapefile.rename(columns=column_replace_dict, inplace=True)
    main_shapefile['geometry'] = main_shapefile['geometry'].apply(lambda geom: geom.wkt if geom else None)
    # main_shapefile['geometry'] = main_shapefile['geometry'].apply(wkt.loads)
    # main_shapefile.set_geometry("geometry", inplace=True)
    # main_shapefile.set_crs("EPSG:3857", inplace=True)
    breakpoint()

    main_shapefile.to_sql('shapefiles', engine, schema='general_data', if_exists='append',index=False)

