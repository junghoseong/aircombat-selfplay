import math

def calculate_coordinates_corrected(center_lat, center_lon, radius_km, angles_deg):
    R = 6371  # 지구 평균 반지름 (km)
    d = radius_km / R  # 구면 거리 (라디안 단위)
    center_lat_rad = math.radians(center_lat)
    center_lon_rad = math.radians(center_lon)
    
    coordinates = []
    for angle_deg in angles_deg:
        # 각도를 아래쪽이 0도로 설정하고 반시계 방향으로 증가
        theta = math.radians(180 - angle_deg)  # 그대로 각도를 사용
        
        # 새로운 위도 계산
        new_lat_rad = math.asin(
            math.sin(center_lat_rad) * math.cos(d) +
            math.cos(center_lat_rad) * math.sin(d) * math.cos(theta)
        )
        
        # 새로운 경도 계산
        new_lon_rad = center_lon_rad + math.atan2(
            math.sin(theta) * math.sin(d) * math.cos(center_lat_rad),
            math.cos(d) - math.sin(center_lat_rad) * math.sin(new_lat_rad)
        )
        
        # 라디안 -> 도(degree) 변환
        new_lat = math.degrees(new_lat_rad)
        new_lon = math.degrees(new_lon_rad)
        
        coordinates.append((new_lat, new_lon))
    
    return coordinates

# 중심점과 반지름
center_lat = 60.1  # 중심 위도
center_lon = 120.0  # 중심 경도
radius_km = 11.119  # 반지름 (위도 0.1도)

# 각도 리스트 (0~360도)
angles_deg = list(range(0, 361, 30))  # 10도 간격

# 좌표 계산 (아래쪽이 0도, 오른쪽이 90도, 위쪽이 180도, 왼쪽이 270도)
result_corrected = calculate_coordinates_corrected(center_lat, center_lon, radius_km, angles_deg)

print(result_corrected)