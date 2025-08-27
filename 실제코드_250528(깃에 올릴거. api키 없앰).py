import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
############################################################

# 코드 실행 되면 울리는 비프음
import winsound as sd
def beepsound():
    fr = 2000    # range : 37 ~ 32767
    du = 1000     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)

############################################################

# 동부 서부 설정
서west_areas = ['매송면', '봉담읍', '향남읍', '양감면', '새솔동', '우정읍', '남양읍', '비봉면', '마도면', '송산면', '서신면', '팔탄면', '장안면']

동east_areas = ['정남면', '기안동', '안녕동', '배양동', '진안동', '병점동', '능동', '기산동', '반월동', '반정동', '황계동', '송산동', '반송동', '석우동', '오산동', '청계동', '영천동', '중동', '신동', '목동', '산척동', '장지동', '송동', '방교동', '금곡동']

len(서west_areas) + len(동east_areas) # 38개

############################################################

# 지도 그리기
import folium

# 지도 생성 (중심은 첫 번째 좌표)
m = folium.Map(location=[37.167104, 126.861176], zoom_start=10, tiles = 'cartodbpositron') # 내가 생각하는 화성시 중심으로 지도 생성

# 경계선 추가
import geopandas as gpd

# 데이터 불러오기
every_emd = gpd.read_file("../data/읍면동/emd.shp", encoding='cp949')

# 행정 코드가 글자임. 숫자로 변경
every_emd['EMD_CD'] = every_emd['EMD_CD'].astype('int')

# 화성시만 추출하기
hscity_emd = every_emd.query("(EMD_CD>=41590000) and (EMD_CD<=41590999)").reset_index(drop=True)

hscity_emd.crs
# 투영 좌표계(Projected CRS)
# PCS_ITRF2000_TM (ITRF2000 기반의 Transverse Mercator)

# 폴리움은 WGS84 (EPSG:4326), 위도경도만 인식함

# 변환
hs_gdf = hscity_emd.to_crs(epsg=4326)


# 동부 서부 나누기
for index in range(len(hs_gdf)):
    if hs_gdf.loc[index, 'EMD_KOR_NM'] in 서west_areas:
        hs_gdf.loc[index, 'west or east side'] = 'west'
    elif hs_gdf.loc[index, 'EMD_KOR_NM'] in 동east_areas:
        hs_gdf.loc[index, 'west or east side'] = 'east'

# 서부만 따로 빼기
west_gdf = hs_gdf[hs_gdf['west or east side'] == "west"].reset_index(drop= True)

# 동부만 따로 빼기
east_gdf = hs_gdf[hs_gdf['west or east side'] == "east"].reset_index(drop= True)

# GeoJSON 형태로 지도에 추가(경계선 추가)

# 서쪽
folium.GeoJson(
    west_gdf.to_json(),
    name='행정경계',
    style_function=lambda x: {
        'fillColor': 'none',
        'color': 'green',
        'weight': 2
    },
    popup=folium.GeoJsonPopup(
        fields=["EMD_KOR_NM"],  # 클릭 시 팝업 내용
        aliases=["클릭한 구역 이름:"],
        localize=True,
        labels=True
    )
).add_to(m)

# 동쪽
folium.GeoJson(
    east_gdf.to_json(),
    name='행정경계',
    style_function=lambda x: {
        'fillColor': 'none',
        'color': 'yellow',
        'weight': 2
    },
    popup=folium.GeoJsonPopup(
        fields=["EMD_KOR_NM"],  # 클릭 시 팝업 내용
        aliases=["클릭한 구역 이름:"],
        localize=True,
        labels=True
    )
).add_to(m)

############################################################

### 건물 점 찍기

#☆★ 데이터 변한 과정 오래걸려서 csv로 저장함. ☆★
'''
# 데이터 불러오기
raw_hsbuilding = pd.read_csv("../data/02. 총괄표제부_20250416141657.csv")

## 내가 쓸 데이터 만들기
mybuilding = raw_hsbuilding[['대지위치', '도로명대지위치', '세대수(세대)', '가구수(가구)']]

# 컬럼 이름 변경
mybuilding.rename(columns={
    '세대수(세대)': '세대수',
    '가구수(가구)': '가구수'
}, inplace=True)

# 세대수랑 가구수가 0인 데이터는 빼기
mybuilding = mybuilding.query("(세대수 > 0) or (가구수 > 0)")
mybuilding.shape #3734 행

# 세대수 + 가구수 더한 컬럼 만들기
mybuilding["거주단위수"] = mybuilding["세대수"] + mybuilding["가구수"]

# 인덱스 재정렬
mybuilding = mybuilding.reset_index(drop = True)

# 위도, 경도 추가
mybuilding["위도"] = np.nan
mybuilding["경도"] = np.nan

'''

## 카카오 API 사용 ##
# 함수 만들기
import requests
def kakao_geocode(address, rest_api_key):
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {
    "Authorization": f"KakaoAK {rest_api_key}"
    }
    params = {
    "query": address
    }

    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    if data['documents']: 
        if data['documents'][0]['address']['region_2depth_name'] == "화성시": # 화성시일 때만 가져오게함.
            result = data['documents'][0]['address']
            lat = result['y']  # 위도
            lon = result['x']  # 경도
            address_name = result['address_name'] # 주소
            return lat, lon, address_name
        else:
            return None
    else:
        return None

    # 설명
    # 주소와 내 rest_api_key를 넣으면, 위도와 경도를 뱉는다.
    # data로 지정한 response.json()는 딕셔너리 형태임.
# 주소와 내 rest_api_key를 넣으면, 위도와 경도를 뱉는다.

# 1초에 30개까지만 변환 되는 것 확인함!!!

rest_api_key = "비밀~~~~~~~~~~~~~~~~~~~~~~~~~" 

'''
# 대지위치로 변환
number_converted = 0
for i in range(len(mybuilding)):
    if not ((np.isnan(mybuilding.loc[i,"위도"])) and (np.isnan(mybuilding.loc[i,"경도"]))):
        continue

    address = mybuilding.loc[i, "대지위치"]
    coords = kakao_geocode(address, rest_api_key)
    if coords is not None:
        mybuilding.loc[i,"위도"] = float(coords[0])
        mybuilding.loc[i,"경도"] = float(coords[1])
        number_converted = number_converted + 1
        print(f"{len(mybuilding)}개 중 {i+1}번째 실행 변환 성공")
    else:
        print(f"{len(mybuilding)}개 중 {i+1}번째 실행 변환 실패")
print("반복문 끝!")
beepsound()
print("변환된 갯수 = " + str(number_converted) + "개")

mybuilding.info() # 대지위치로만 변환한 것 = 3242 개 변환됨
# 대지위치로만 변환할 때 안 되는것들이 있다!

# 도로명대지위치를 이용하여 변환해보자!
number_converted = 0
for i in range(len(mybuilding)):
    if not ((np.isnan(mybuilding.loc[i,"위도"])) and (np.isnan(mybuilding.loc[i,"경도"]))):
        continue

    address = mybuilding.loc[i, "도로명대지위치"]
    coords = kakao_geocode(address, rest_api_key)
    if coords is not None:
        mybuilding.loc[i,"위도"] = float(coords[0])
        mybuilding.loc[i,"경도"] = float(coords[1])
        number_converted = number_converted + 1
        print(f"{len(mybuilding)}개 중 {i+1}번째 실행 변환 성공")
    else:
        print(f"{len(mybuilding)}개 중 {i+1}번째 실행 변환 실패")
print("반복문 끝!")
beepsound()
print("변환된 갯수 = " + str(number_converted) + "개")
# 21개 변환됨
mybuilding.info() # 3263 개 변환됨

# 변환된것만 데이터프레임으로 쓰기
변환되지_않은_행 =np.isnan(mybuilding.loc[:, "위도"])
추출할_인덱스 = ~변환되지_않은_행
converted_building = mybuilding.loc[추출할_인덱스].reset_index(drop=True)
sum(np.isnan(converted_building.loc[:, "위도"])) # 확인

converted_building.to_csv("converted_building.csv", encoding = 'cp949', index=False)
'''
#☆★ 데이터 변한 과정 오래걸려서 csv로 저장함. ☆★
converted_building = pd.read_csv("converted_building.csv", encoding = 'cp949')


# 동부 서부 나누기

# 읍면동 컬럼 추가
converted_building['읍면동'] = ""
for index in range(len(converted_building)):
    if converted_building['대지위치'].isnull()[index] == True:
        continue
    a = converted_building['대지위치'][index].split(' ')
    converted_building['읍면동'][index] = a[2]

# 동부 서부 나누기
for index in range(len(converted_building)):
    if converted_building.loc[index, '읍면동'] in 서west_areas:
        converted_building.loc[index, '서부or동부'] = '서부'
    elif converted_building.loc[index, '읍면동'] in 동east_areas:
        converted_building.loc[index, '서부or동부'] = '동부'




# 건물 점 찍기
for _, row in converted_building.iterrows():
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=2,               # 점 크기
        color=None,            # 테두리 없음
        fill=True,
        fill_color='blue',       # 내부 색
        fill_opacity=0.8,
        popup=f"주소:\
            <br>{row['대지위치']}\
            <br>위도: {row['위도']}\
            <br>경도: {row['경도']}\
            <br>거주단위수: {row['거주단위수']}"
    ).add_to(m)


# 지도 저장 (또는 Jupyter 노트북에서는 바로 표시 가능)
# m.save('map.html')  # 파일로 저장
# m  # 노트북에서 바로 렌더링

#####

# 화성시 이미 있는 체육관 csv 만들기
#☆★ 데이터 변한 과정 오래걸려서 csv로 저장함. ☆★
'''
raw_gym = pd.read_csv("../data/화성도시공사_화성시 체육시설 현황_20241231.csv", encoding = 'cp949')

# 시설명을 하나의 문자열로 합치는 함수
def 시설명_합치기(시설_목록):
    시설_중복제거 = set(시설_목록)
    시설_정렬 = sorted(시설_중복제거)
    결과 = ', '.join(시설_정렬)
    return 결과

# 그룹화 및 시설명 합치기
gym = raw_gym.groupby(['공원명', '위치', '총 면적'])['시설명'].apply(시설명_합치기).reset_index()
# 135 rows × 4 columns

# 컬럼 이름 변경
gym.rename(columns= {'공원명' : '장소명'}, inplace= True)

# 대지위치 위도 경도 추가
gym["대지위치"] = np.nan
gym["위도"] = np.nan
gym["경도"] = np.nan


number_converted = 0
for i in range(len(gym)):
    if not ((np.isnan(gym.loc[i,"위도"])) and (np.isnan(gym.loc[i,"경도"]))):
        continue

    address = gym.loc[i, "위치"]
    coords = kakao_geocode(address, rest_api_key)
    if coords is not None:
        gym.loc[i,"위도"] = float(coords[0])
        gym.loc[i,"경도"] = float(coords[1])
        gym.loc[i,"대지위치"] = coords[2]
        number_converted = number_converted + 1
        print(f"{len(gym)} 번째 실행 중 {i+1}번째 실행 변환 성공")
    else:
        print(f"{len(gym)} 번째 실행 중 {i+1}번째 실행 변환 실패")
print("반복문 끝!")
beepsound()
print("변환된 갯수 = " + str(number_converted) + "개") # 106개

# 변환된것만 데이터프레임으로 쓰기
변환되지_않은_행 =np.isnan(gym.loc[:, "위도"])
추출할_인덱스 = ~변환되지_않은_행
converted_gym = gym.loc[추출할_인덱스].reset_index(drop=True)
sum(np.isnan(converted_gym.loc[:, "위도"])) # 확인

converted_gym.to_csv("converted_gym.csv", encoding = 'cp949', index=False)
'''
#☆★ 데이터 변한 과정 오래걸려서 csv로 저장함. ☆★

converted_gym = pd.read_csv("converted_gym.csv", encoding = 'cp949')

# 읍면동 컬럼 추가
converted_gym['읍면동'] = ""

for index in range(len(converted_gym)):
    if converted_gym["대지위치"].isnull()[index] == True:
        continue
    a = converted_gym["대지위치"][index].split(' ')
    converted_gym['읍면동'][index] = a[2]

# 동부 서부 나누기
for index in range(len(converted_gym)):
    if converted_gym.loc[index, '읍면동'] in 서west_areas:
        converted_gym.loc[index, '서부or동부'] = '서부'
    elif converted_gym.loc[index, '읍면동'] in 동east_areas:
        converted_gym.loc[index, '서부or동부'] = '동부'



# 점수 추가
converted_gym['시설 점수'] = converted_gym['총 면적'] // 100
# 총 면적 100당 1점으로 설정.

converted_gym['시설 점수'] = converted_gym['시설 점수'].astype('int')

# 지도에 표시하기
for _, row in converted_gym.iterrows():
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=2,               # 점 크기
        color=None,            # 테두리 없음
        fill=True,
        fill_color='red',       # 내부 색
        fill_opacity=0.8,
        popup=f"장소명:\
            <br>{row['장소명']}\
            <br>시설 종류:\
            <br>{row['시설명']}\
            <br>위도:\
            <br>{row['위도']}\
            <br>경도:\
            <br>{row['경도']}\
            <br>시설 점수: {row['시설 점수']}"
    ).add_to(m)

# m.save('map.html')  # 파일로 저장


# 최적의 점 찾기

# 그리드를 그린 뒤 그리드의 각 점이 중심인 원을 그림.
# 각 점마다 계산
# 계산 식: 1. 몇 m 이내에 {각 건물마다 거주단위수} 모두 더함
#         2. 같은 범위 내에 {각 체육시설마다 점수} 모두 더함
#         3. 1-2 = 점수

from geopy.distance import distance
from shapely.geometry import Point
#☆★ 계산 과정 오래걸려서 csv로 저장함. ☆★


간격 = 0.01 # 약 1000m

'''
# 위경도 범위 추출
minx, miny, maxx, maxy = hs_gdf.total_bounds # 화성시 경계선 기준으로 범위 추출함.

# 그리드 생성
x_coords = np.arange(minx, maxx, 간격)
y_coords = np.arange(miny, maxy, 간격)

# point 객체 생성
points = []

# x 좌표 반복
for x in x_coords:
    # y 좌표 반복
    for y in y_coords:
        # (x, y) 좌표에 해당하는 점을 만든다
        p = Point(x, y)
        # 리스트에 추가
        points.append(p)

# 점들을 GeoDataFrame으로 변환
points_gdf = gpd.GeoDataFrame(geometry=points, crs=hs_gdf.crs)

# 포함 여부 판단 (화성시 경계 안인지 여부)
points_gdf["hs_inside"] = points_gdf.within(hs_gdf.unary_union)

# True/False로 필터링 가능
hs_inside_points = points_gdf[points_gdf["hs_inside"]].reset_index(drop=True)
hs_outside_points = points_gdf[~points_gdf["hs_inside"]].reset_index(drop=True)


# 이걸 또 서부, 동부로 나누기
points_gdf["west_inside"] = points_gdf.within(west_gdf.unary_union)

points_gdf["east_inside"] = points_gdf.within(east_gdf.unary_union)

west_inside_points = points_gdf[points_gdf["west_inside"]].reset_index(drop=True)

east_inside_points = points_gdf[points_gdf["east_inside"]].reset_index(drop=True)

# 결과 저장용 데이터프레임
hs_df_grid_score = gpd.GeoDataFrame(columns=['geometry', '그리드_넘버','위도', '경도', '그리드_점수'],
                                 index=range(len(hs_inside_points)))
hs_df_grid_score.geometry = hs_inside_points.geometry.reset_index(drop=True)

west_df_grid_score = gpd.GeoDataFrame(columns=['geometry', '그리드_넘버','위도', '경도', '그리드_점수'],
                                 index=range(len(west_inside_points)))
west_df_grid_score.geometry = west_inside_points.geometry.reset_index(drop=True)

east_df_grid_score = gpd.GeoDataFrame(columns=['geometry', '그리드_넘버','위도', '경도', '그리드_점수'],
                                 index=range(len(east_inside_points)))
east_df_grid_score.geometry = east_inside_points.geometry.reset_index(drop=True)
'''


# 반복문 실행
radius = 1200 # 원의 반지름 (미터 단위)

'''
def 임의의_점_점수계산_함수(dataframe):
    for index in range(len(dataframe)):
        dataframe.loc[index, '그리드_넘버'] = index
        dataframe.loc[index, '위도'] = dataframe.geometry.y.iloc[index]
        dataframe.loc[index, '경도'] = dataframe.geometry.x.iloc[index]

        center = (dataframe.geometry.y.iloc[index], dataframe.geometry.x.iloc[index]) # (위도, 경도)

        # 건물 점수 계산
        plus_score = 0
        for i in range(len(converted_building)):
            point = (converted_building.loc[i, '위도'], converted_building.loc[i, '경도'])
            dist = distance(center, point).meters
            if dist <= radius:
                plus_score += converted_building.loc[i, '거주단위수']
        # 체육시설 점수 계산
        minus_score = 0
        for i in range(len(converted_gym)):
            point = (converted_gym.loc[i, '위도'], converted_gym.loc[i, '경도'])
            dist = distance(center, point).meters
            if dist <= radius:
                minus_score += converted_gym.loc[i, '시설 점수']
        dataframe.loc[index, '그리드_점수'] = plus_score - minus_score
        print(f"총 {len(dataframe)}개의 점 중에 {index + 1}번째 점 계산 완료")
    print("함수 실행 끝!")
    beepsound()


# 계산하기
임의의_점_점수계산_함수(hs_df_grid_score) # 698개 점에 1200m 안에 것들로 계산했을 때 6분 걸림

임의의_점_점수계산_함수(west_df_grid_score) # 568개의 점에 1200m 안에 것들로 계산했을 때 5분 걸림

임의의_점_점수계산_함수(east_df_grid_score) # 130개의 점에 1200m 안에 것들로 계산했을 때 1분

# 확인
hs_df_grid_score

west_df_grid_score

east_df_grid_score


### csv로 저장
hs_df_grid_score.to_csv("hs df grid score_0.01_1200m.csv", encoding = 'cp949', index=False)
# 0.01은 간격을 뜻함.
# 1200m는 원의 반지름을 뜻함.

west_df_grid_score.to_csv("west df grid score_0.01_1200m.csv", encoding = 'cp949', index=False)

east_df_grid_score.to_csv("east df grid score_0.01_1200m.csv", encoding = 'cp949', index=False)
'''

#☆★ 계산 과정 오래걸려서 csv로 저장함. ☆★
west_df_grid_score = pd.read_csv("west df grid score_0.01_1200m.csv", encoding = 'cp949')
east_df_grid_score = pd.read_csv("east df grid score_0.01_1200m.csv", encoding = 'cp949')


# 점수가 가장 높은 그리드 점 찾기!
def find_best_point(dataframe, n):
    best_df = dataframe.sort_values('그리드_점수', ascending= False).reset_index(drop=True)
    best_df['순위'] = None
    for i in range(n):
        best_df.loc[i, '순위'] = i+1
        선택된_점의_위도 = best_df.loc[i, '위도']
        선택된_점의_경도 = best_df.loc[i, '경도']
        선택된_점의_그리드_넘버 = best_df.loc[i, '그리드_넘버']
        center = (선택된_점의_위도, 선택된_점의_경도)
        삭제할_그리드_넘버 = list()
        for index in range(len(best_df)):
            point = (best_df.loc[index, '위도'], best_df.loc[index, '경도'])
            dist = distance(center, point).meters
            num = best_df.loc[index, '그리드_넘버']
            if point == center:
                continue
            elif dist <= radius*2:
                삭제할_그리드_넘버.append(num)
        print(f"{선택된_점의_그리드_넘버}번 점에 포함되어 삭제되는 점은 {삭제할_그리드_넘버}녀석들이다.")
        drop_index = best_df.query("그리드_넘버 in @삭제할_그리드_넘버").index
        best_df = best_df.drop(drop_index, axis=0).reset_index(drop=True)
    return best_df.head(n)

west_best = find_best_point(west_df_grid_score, 5)
east_best = find_best_point(east_df_grid_score, 5)


# 추천하는 점 찍기
def best_point_draw(dataframe, mycolor):
    for index in range(len(dataframe)):
        해당_행_위도 = dataframe.loc[index, '위도']
        해당_행_경도 = dataframe.loc[index, '경도']
        해당_행_점수 = dataframe.loc[index, '그리드_점수']
        해당_행_순위 = dataframe.loc[index, '순위']
        center = (해당_행_위도, 해당_행_경도)

        # 지도 위에 원, 점 추가
        folium.Circle(
            location=center,
            radius=radius,
            color=f"{mycolor}"
        ).add_to(m)  

        folium.Marker(
            location=center,
            icon=folium.DivIcon(
                icon_size=(30, 30),            # HTML box 크기
                icon_anchor=(15, 15),          # (가로/세로 중심 정렬)
                html=f"""
                <div style="
                    background-color: {mycolor};
                    border-radius: 50%;
                    color: white;
                    text-align: center;
                    font-size: 14px;
                    font-weight: bold;
                    height: 30px;
                    width: 30px;
                    line-height: 30px;
                    border: 2px solid white;
                    box-shadow: 1px 1px 2px gray;
                ">
                    {해당_행_순위}
                </div>
                """
            ),
            popup=f"추천 위치<br>\
                순위: {해당_행_순위}<br>\
                위도: {해당_행_위도}<br>\
                경도: {해당_행_경도}<br>\
                총 점수: {해당_행_점수}"
        ).add_to(m)

    # border-radius: 50%	원형 만들기
    # height/width: 30px	원의 크기
    # line-height: 30px	텍스트를 수직 중앙 정렬
    # font-size, font-weight	글자 크기 및 굵기
    # background-color	원의 배경색
    # color	글자 색
    # border, box-shadow	테두리와 그림자 효과

# 서부 점 찍기
best_point_draw(west_best, "orange")

# 동부 점 찍기
best_point_draw(east_best, "skyblue")

# 저장
m.save('map.html')


############################################################
