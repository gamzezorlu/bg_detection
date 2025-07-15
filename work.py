import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Doğalgaz Kaçak Kullanım Tespit Sistemi",
    page_icon="🔍",
    layout="wide"
)

# Ana başlık
st.title("🔍 Doğalgaz Kaçak Kullanım Tespit Sistemi")
st.markdown("**Gelişmiş Anomali Tespit Algoritması**")
st.markdown("---")

# Sidebar - Dosya yükleme
st.sidebar.header("📁 Dosya Yükleme")
uploaded_file = st.sidebar.file_uploader(
    "Excel dosyasını yükleyin",
    type=['xlsx', 'xls'],
    help="Tesisat no, tarih, bağlantı nesnesi ve tüketim (sm3) sütunlarını içeren Excel dosyası"
)

# Analiz parametreleri
st.sidebar.header("⚙️ Analiz Parametreleri")
winter_drop_threshold = st.sidebar.slider(
    "Kış Ayı Düşüş Eşiği (%)",
    min_value=20,
    max_value=80,
    value=50,
    help="Önceki kış aylarına göre düşüş yüzdesi"
)

building_avg_threshold = st.sidebar.slider(
    "Bina Ortalaması Sapma Eşiği (%)",
    min_value=20,
    max_value=80,
    value=40,
    help="Bina ortalamasından sapma yüzdesi"
)

pattern_deviation_threshold = st.sidebar.slider(
    "Genel Örüntü Sapma Eşiği",
    min_value=0.1,
    max_value=0.5,
    value=0.3,
    help="Genel tüketim örüntüsünden sapma eşiği"
)

min_months = st.sidebar.slider(
    "Minimum Veri Süresi (Ay)",
    min_value=12,
    max_value=36,
    value=18,
    help="Analiz için minimum veri süresi"
)

def load_and_process_data(file):
    """Veriyi yükle ve işle"""
    try:
        df = pd.read_excel(file)
        df.columns = df.columns.str.lower().str.strip()
        
        # Sütun adlarını kontrol et ve düzelt
        column_mapping = {
            'tesisat no': 'tesisat_no',
            'tesisatno': 'tesisat_no',
            'baglanti nesnesi': 'baglanti_nesnesi',
            'baglantinesnesi': 'baglanti_nesnesi',
            'bağlantı nesnesi': 'baglanti_nesnesi',
            'bağlantınesnesi': 'baglanti_nesnesi'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        required_cols = ['tesisat_no', 'tarih', 'baglanti_nesnesi', 'sm3']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Eksik sütunlar: {missing_cols}")
            st.info("Mevcut sütunlar: " + ", ".join(df.columns.tolist()))
            return None
        
        # Veri tiplerini düzelt
        df['tarih'] = pd.to_datetime(df['tarih'])
        df['sm3'] = pd.to_numeric(df['sm3'], errors='coerce')
        df['tesisat_no'] = pd.to_numeric(df['tesisat_no'], errors='coerce')
        df['baglanti_nesnesi'] = pd.to_numeric(df['baglanti_nesnesi'], errors='coerce')
        
        # Eksik verileri temizle
        df = df.dropna(subset=['tesisat_no', 'tarih', 'sm3', 'baglanti_nesnesi'])
        
        # Tarih bilgilerini ekle
        df['yil'] = df['tarih'].dt.year
        df['ay'] = df['tarih'].dt.month
        df['mevsim'] = df['ay'].apply(lambda x: 'Kış' if x in [12, 1, 2] else 
                                               'İlkbahar' if x in [3, 4, 5] else 
                                               'Yaz' if x in [6, 7, 8] else 'Sonbahar')
        
        # Kış ayları flagı
        df['is_winter'] = df['ay'].isin([12, 1, 2])
        
        return df
    
    except Exception as e:
        st.error(f"Veri yükleme hatası: {str(e)}")
        return None

def calculate_building_averages(df):
    """Bina bazında ortalamalar hesapla"""
    building_stats = df.groupby(['baglanti_nesnesi', 'ay']).agg({
        'sm3': ['mean', 'std', 'count']
    }).reset_index()
    
    building_stats.columns = ['baglanti_nesnesi', 'ay', 'bina_ortalama', 'bina_std', 'tesisat_sayisi']
    
    # Aylık genel ortalamalar
    monthly_avg = df.groupby('ay')['sm3'].mean().reset_index()
    monthly_avg.columns = ['ay', 'genel_ortalama']
    
    return building_stats, monthly_avg

def detect_consumption_pattern(df):
    """Genel tüketim örüntüsünü tespit et"""
    # Aylık normalize edilmiş tüketim örüntüsü
    monthly_pattern = df.groupby('ay')['sm3'].mean()
    monthly_pattern = monthly_pattern / monthly_pattern.max()  # Normalize et
    
    # Seasonal decomposition benzeri yaklaşım
    seasonal_component = {}
    for month in range(1, 13):
        seasonal_component[month] = monthly_pattern.get(month, 0)
    
    return seasonal_component

def calculate_pattern_deviation(facility_data, seasonal_pattern):
    """Tesisatın genel örüntüden sapmasını hesapla"""
    if len(facility_data) < 12:
        return 0
    
    # Tesisatın aylık ortalamalarını hesapla
    facility_monthly = facility_data.groupby('ay')['sm3'].mean()
    
    # Normalize et
    if facility_monthly.max() > 0:
        facility_monthly = facility_monthly / facility_monthly.max()
    
    # Örüntü sapmasını hesapla
    deviations = []
    for month in range(1, 13):
        if month in facility_monthly.index:
            expected = seasonal_pattern.get(month, 0)
            actual = facility_monthly.get(month, 0)
            if expected > 0:
                deviation = abs(actual - expected) / expected
                deviations.append(deviation)
    
    return np.mean(deviations) if deviations else 0

def detect_winter_consumption_drop(facility_data):
    """Kış aylarında ani düşüş tespit et"""
    winter_data = facility_data[facility_data['is_winter']].copy()
    
    if len(winter_data) < 6:  # En az 2 kış sezonu
        return False, 0, None
    
    winter_data = winter_data.sort_values('tarih')
    
    # Yıllık kış ortalamaları
    winter_yearly = winter_data.groupby('yil')['sm3'].mean().reset_index()
    
    if len(winter_yearly) < 2:
        return False, 0, None
    
    # Son yıl ile önceki yılları karşılaştır
    last_year = winter_yearly['yil'].max()
    previous_years = winter_yearly[winter_yearly['yil'] < last_year]
    
    if len(previous_years) == 0:
        return False, 0, None
    
    last_year_avg = winter_yearly[winter_yearly['yil'] == last_year]['sm3'].iloc[0]
    previous_avg = previous_years['sm3'].mean()
    
    if previous_avg > 0:
        drop_percentage = (previous_avg - last_year_avg) / previous_avg * 100
        
        # Ani düşüş kontrolü - ardışık aylarda düşüş
        recent_winter = winter_data[winter_data['yil'] >= last_year - 1].sort_values('tarih')
        
        if len(recent_winter) >= 3:
            # Son 3 kış ayının ortalaması ile önceki dönem karşılaştırması
            if len(recent_winter) >= 6:
                mid_point = len(recent_winter) // 2
                recent_avg = recent_winter.iloc[mid_point:]['sm3'].mean()
                older_avg = recent_winter.iloc[:mid_point]['sm3'].mean()
                
                if older_avg > 0:
                    additional_drop = (older_avg - recent_avg) / older_avg * 100
                    drop_percentage = max(drop_percentage, additional_drop)
        
        return drop_percentage > 30, drop_percentage, {
            'last_year_avg': last_year_avg,
            'previous_avg': previous_avg,
            'drop_date': winter_data[winter_data['yil'] == last_year]['tarih'].min()
        }
    
    return False, 0, None

def detect_building_average_deviation(facility_data, building_stats):
    """Bina ortalamasından sapma tespit et"""
    if facility_data.empty:
        return False, 0, None
    
    building_id = facility_data['baglanti_nesnesi'].iloc[0]
    
    # Tesisatın aylık ortalamaları
    facility_monthly = facility_data.groupby('ay')['sm3'].mean()
    
    # Bina ortalamaları
    building_monthly = building_stats[building_stats['baglanti_nesnesi'] == building_id]
    
    if building_monthly.empty:
        return False, 0, None
    
    deviations = []
    significant_deviations = []
    
    for month in range(1, 13):
        if month in facility_monthly.index:
            facility_consumption = facility_monthly[month]
            building_avg_row = building_monthly[building_monthly['ay'] == month]
            
            if not building_avg_row.empty:
                building_avg = building_avg_row['bina_ortalama'].iloc[0]
                
                if building_avg > 0:
                    deviation_pct = (building_avg - facility_consumption) / building_avg * 100
                    deviations.append(deviation_pct)
                    
                    if deviation_pct > 30:  # %30'dan fazla düşükse
                        significant_deviations.append({
                            'month': month,
                            'facility_consumption': facility_consumption,
                            'building_avg': building_avg,
                            'deviation_pct': deviation_pct
                        })
    
    avg_deviation = np.mean(deviations) if deviations else 0
    
    return len(significant_deviations) >= 3, avg_deviation, significant_deviations

def advanced_anomaly_detection(df, winter_drop_threshold, building_avg_threshold, pattern_deviation_threshold, min_months):
    """Gelişmiş anomali tespit algoritması"""
    
    # Bina ortalamaları hesapla
    building_stats, monthly_avg = calculate_building_averages(df)
    
    # Genel tüketim örüntüsünü tespit et
    seasonal_pattern = detect_consumption_pattern(df)
    
    suspicious_facilities = []
    
    # Her tesisat için analiz
    for tesisat in df['tesisat_no'].unique():
        facility_data = df[df['tesisat_no'] == tesisat].copy()
        facility_data = facility_data.sort_values('tarih')
        
        # Minimum veri kontrolü
        if len(facility_data) < min_months:
            continue
        
        anomalies = []
        risk_factors = []
        
        # 1. Kış aylarında ani düşüş kontrolü
        has_winter_drop, winter_drop_pct, winter_details = detect_winter_consumption_drop(facility_data)
        
        if has_winter_drop and winter_drop_pct > winter_drop_threshold:
            anomalies.append({
                'type': 'Kış Ayı Ani Düşüş',
                'severity': 'Yüksek',
                'details': f"Önceki kış aylarına göre %{winter_drop_pct:.1f} düşüş",
                'value': winter_drop_pct,
                'date': winter_details['drop_date'] if winter_details else None
            })
            risk_factors.append(winter_drop_pct)
        
        # 2. Bina ortalaması sapma kontrolü
        has_building_deviation, building_deviation_pct, building_details = detect_building_average_deviation(
            facility_data, building_stats
        )
        
        if has_building_deviation and building_deviation_pct > building_avg_threshold:
            anomalies.append({
                'type': 'Bina Ortalaması Altında Tüketim',
                'severity': 'Orta',
                'details': f"Bina ortalamasından %{building_deviation_pct:.1f} düşük",
                'value': building_deviation_pct,
                'monthly_details': building_details
            })
            risk_factors.append(building_deviation_pct * 0.7)  # Biraz daha düşük ağırlık
        
        # 3. Genel örüntü sapma kontrolü
        pattern_deviation = calculate_pattern_deviation(facility_data, seasonal_pattern)
        
        if pattern_deviation > pattern_deviation_threshold:
            anomalies.append({
                'type': 'Genel Tüketim Örüntüsü Sapması',
                'severity': 'Orta',
                'details': f"Genel örüntüden {pattern_deviation:.2f} sapma",
                'value': pattern_deviation * 100,
                'seasonal_pattern': seasonal_pattern
            })
            risk_factors.append(pattern_deviation * 100)
        
        # 4. İstatistiksel anomali tespit (Isolation Forest)
        if len(facility_data) >= 24:  # En az 2 yıl veri
            try:
                # Özellik çıkarımı
                features = []
                for _, row in facility_data.iterrows():
                    features.append([
                        row['sm3'],
                        row['ay'],
                        row['yil'],
                        1 if row['is_winter'] else 0
                    ])
                
                features = np.array(features)
                
                if len(features) > 10:
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    anomaly_scores = iso_forest.fit_predict(features)
                    
                    anomaly_count = np.sum(anomaly_scores == -1)
                    anomaly_ratio = anomaly_count / len(anomaly_scores)
                    
                    if anomaly_ratio > 0.15:  # %15'den fazla anomali
                        anomalies.append({
                            'type': 'İstatistiksel Anomali',
                            'severity': 'Düşük',
                            'details': f"Verinin %{anomaly_ratio*100:.1f}'i anomali",
                            'value': anomaly_ratio * 100
                        })
                        risk_factors.append(anomaly_ratio * 50)
            except:
                pass
        
        # 5. Sürekli düşük tüketim (kış ayları için)
        winter_data = facility_data[facility_data['is_winter']]
        if len(winter_data) > 6:
            winter_avg = winter_data['sm3'].mean()
            general_winter_avg = df[df['is_winter']]['sm3'].mean()
            
            if winter_avg < general_winter_avg * 0.3:  # Genel ortalamanın %30'u altında
                anomalies.append({
                    'type': 'Sürekli Düşük Kış Tüketimi',
                    'severity': 'Yüksek',
                    'details': f"Genel kış ortalamasının %{(winter_avg/general_winter_avg)*100:.1f}'i",
                    'value': (general_winter_avg - winter_avg) / general_winter_avg * 100
                })
                risk_factors.append(70)
        
        # Şüpheli tesisatları kaydet
        if anomalies:
            # Risk skoru hesapla
            base_risk = len(anomalies) * 20
            factor_risk = sum(risk_factors) * 0.5
            risk_score = base_risk + factor_risk
            
            # Severity ağırlıkları
            severity_weights = {'Yüksek': 1.5, 'Orta': 1.0, 'Düşük': 0.5}
            severity_multiplier = sum([severity_weights.get(a['severity'], 1.0) for a in anomalies])
            risk_score *= severity_multiplier
            
            suspicious_facilities.append({
                'tesisat_no': tesisat,
                'baglanti_nesnesi': facility_data['baglanti_nesnesi'].iloc[0],
                'toplam_anomali': len(anomalies),
                'risk_skoru': risk_score,
                'yuksek_risk_anomali': len([a for a in anomalies if a['severity'] == 'Yüksek']),
                'orta_risk_anomali': len([a for a in anomalies if a['severity'] == 'Orta']),
                'dusuk_risk_anomali': len([a for a in anomalies if a['severity'] == 'Düşük']),
                'anomali_tipleri': ', '.join([a['type'] for a in anomalies]),
                'ortalama_tuketim': facility_data['sm3'].mean(),
                'kis_ortalama': facility_data[facility_data['is_winter']]['sm3'].mean() if len(facility_data[facility_data['is_winter']]) > 0 else 0,
                'yaz_ortalama': facility_data[facility_data['ay'].isin([6,7,8])]['sm3'].mean() if len(facility_data[facility_data['ay'].isin([6,7,8])]) > 0 else 0,
                'son_tuketim': facility_data['sm3'].iloc[-1],
                'ilk_anomali_tarihi': min([a['date'] for a in anomalies if a.get('date')] + [facility_data['tarih'].min()]),
                'anomali_detaylari': anomalies
            })
    
    return pd.DataFrame(suspicious_facilities), seasonal_pattern, building_stats

def create_advanced_visualizations(df, suspicious_df, seasonal_pattern, building_stats):
    """Gelişmiş görselleştirmeler"""
    
    # 1. Genel tüketim örüntüsü
    pattern_df = pd.DataFrame(list(seasonal_pattern.items()), columns=['Ay', 'Normalize_Tuketim'])
    pattern_df['Ay_Adi'] = pattern_df['Ay'].map({
        1: 'Oca', 2: 'Şub', 3: 'Mar', 4: 'Nis', 5: 'May', 6: 'Haz',
        7: 'Tem', 8: 'Ağu', 9: 'Eyl', 10: 'Eki', 11: 'Kas', 12: 'Ara'
    })
    
    fig1 = px.line(pattern_df, x='Ay_Adi', y='Normalize_Tuketim', 
                   title='Genel Tüketim Örüntüsü (Normalize Edilmiş)',
                   labels={'Normalize_Tuketim': 'Normalize Tüketim', 'Ay_Adi': 'Ay'})
    fig1.update_layout(height=400)
    
    # 2. Risk kategorisi dağılımı
    if not suspicious_df.empty:
        risk_categories = []
        for _, row in suspicious_df.iterrows():
            if row['risk_skoru'] > 200:
                risk_categories.append('Çok Yüksek Risk')
            elif row['risk_skoru'] > 100:
                risk_categories.append('Yüksek Risk')
            elif row['risk_skoru'] > 50:
                risk_categories.append('Orta Risk')
            else:
                risk_categories.append('Düşük Risk')
        
        risk_df = pd.DataFrame({'Risk_Kategori': risk_categories})
        risk_count = risk_df['Risk_Kategori'].value_counts().reset_index()
        risk_count.columns = ['Risk_Kategori', 'Sayi']
        
        fig2 = px.pie(risk_count, values='Sayi', names='Risk_Kategori',
                      title='Risk Kategorisi Dağılımı')
        fig2.update_layout(height=400)
    else:
        fig2 = go.Figure()
        fig2.add_annotation(text="Şüpheli tesisat bulunamadı", x=0.5, y=0.5)
        fig2.update_layout(height=400, title="Risk Kategorisi Dağılımı")
    
    # 3. Anomali tip dağılımı
    if not suspicious_df.empty:
        anomaly_types = []
        for _, row in suspicious_df.iterrows():
            for anomaly in row['anomali_detaylari']:
                anomaly_types.append(anomaly['type'])
        
        anomaly_df = pd.DataFrame({'Anomali_Tip': anomaly_types})
        anomaly_count = anomaly_df['Anomali_Tip'].value_counts().reset_index()
        anomaly_count.columns = ['Anomali_Tip', 'Sayi']
        
        fig3 = px.bar(anomaly_count, x='Anomali_Tip', y='Sayi',
                      title='Anomali Tiplerinin Dağılımı')
        fig3.update_layout(height=400, xaxis_tickangle=-45)
    else:
        fig3 = go.Figure()
        fig3.add_annotation(text="Şüpheli tesisat bulunamadı", x=0.5, y=0.5)
        fig3.update_layout(height=400, title="Anomali Tiplerinin Dağılımı")
    
    # 4. Kış-Yaz tüketim karşılaştırması
    if not suspicious_df.empty:
        fig4 = px.scatter(suspicious_df, x='yaz_ortalama', y='kis_ortalama',
                         size='risk_skoru', color='toplam_anomali',
                         title='Kış-Yaz Tüketim Karşılaştırması',
                         labels={'yaz_ortalama': 'Yaz Ortalama (sm³)', 
                                'kis_ortalama': 'Kış Ortalama (sm³)'})
        fig4.update_layout(height=400)
    else:
        fig4 = go.Figure()
        fig4.add_annotation(text="Şüpheli tesisat bulunamadı", x=0.5, y=0.5)
        fig4.update_layout(height=400, title="Kış-Yaz Tüketim Karşılaştırması")
    
    return fig1, fig2, fig3, fig4

def export_advanced_results(suspicious_df, df, seasonal_pattern, building_stats):
    """Gelişmiş sonuçları Excel formatında hazırla"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # 1. Şüpheli tesisatlar özeti
        if not suspicious_df.empty:
            export_df = suspicious_df[['tesisat_no', 'baglanti_nesnesi', 'risk_skoru', 
                                     'yuksek_risk_anomali', 'orta_risk_anomali', 
                                     'dusuk_risk_anomali', 'anomali_tipleri', 
                                     'ortalama_tuketim', 'kis_ortalama', 'yaz_ortalama',
                                     'son_tuketim', 'ilk_anomali_tarihi']].copy()
            
            export_df['risk_skoru'] = export_df['risk_skoru'].round(2)
            export_df['ortalama_tuketim'] = export_df['ortalama_tuketim'].round(2)
            export_df['kis_ortalama'] = export_df['kis_ortalama'].round(2)
            export_df['yaz_ortalama'] = export_df['yaz_ortalama'].round(2)
            
            export_df.to_excel(writer, sheet_name='Şüpheli Tesisatlar', index=False)
            
            # 2. Detaylı anomali bilgileri
            detail_rows = []
            for _, row in suspicious_df.iterrows():
                for anomaly in row['anomali_detaylari']:
                    detail_rows.append({
                        'tesisat_no': row['tesisat_no'],
                        'baglanti_nesnesi': row['baglanti_nesnesi'],
                        'anomali_tipi': anomaly['type'],
                        'oncelik': anomaly['severity'],
                        'aciklama': anomaly['details'],
                        'deger': anomaly.get('value', 0),
                        'tarih': anomaly.get('date', '')
                    })
            
            detail_df = pd.DataFrame(detail_rows)
            detail_df.to_excel(writer, sheet_name='Anomali Detayları', index=False)
        
        # 3. Genel istatistikler
        total_facilities = df['tesisat_no'].nunique()
        suspicious_count = len(suspicious_df) if not suspicious_df.empty else 0
        
        stats_df = pd.DataFrame({
            'Metrik': [
                'Toplam Tesisat Sayısı',
                'Şüpheli Tesisat Sayısı',
                'Şüpheli Oran (%)',
                'Yüksek Risk Tesisat',
                'Orta Risk Tesisat',
                'Düşük Risk Tesisat',
                'Ortalama Risk Skoru',
                'Analiz Edilen Dönem',
                'En Sık Anomali Tipi'
            ],
            'Değer': [
                total_facilities,
                suspicious_count,
                round(suspicious_count / total_facilities * 100, 2) if total_facilities > 0 else 0,
                len(suspicious_df[suspicious_df['risk_skoru'] > 200]) if not suspicious_df.empty else 0,
                len(suspicious_df[(suspicious_df['risk_skoru'] > 100) & (suspicious_df['risk_skoru'] <= 200)]) if not suspicious_df.empty else 0,
                len(suspicious_df[suspicious_df['risk_skoru'] <= 100]) if not suspicious_df.empty else 0,
                round(suspicious_df['risk_skoru'].mean(), 2) if not suspicious_df.empty else 0,
                f"{df['tarih'].min().strftime('%Y-%m')} - {df['tarih'].max().strftime('%Y-%m')}",
                suspicious_df['anomali_tipleri'].value_counts().index[0] if not suspicious_df.empty else 'Yok'
            ]
        })
        
        stats_df.to_excel(writer, sheet_name='Genel İstatistikler', index=False)
        
        # 4. Bina bazında analiz
        if not building_stats.empty:
            building_summary = building_stats.groupby('baglanti_nesnesi').agg({
                'bina_ortalama': 'mean',
                'tesisat_sayisi': 'mean'
            }).reset_index()
            
            # Şüpheli tesisatların bina dağılımı
            if not suspicious_df.empty:
                suspicious_buildings = suspicious_df['baglanti_nesnesi'].value_counts().reset_index()
                suspicious_buildings.columns = ['baglanti_nesnesi', 'suheli_tesisat_sayisi']
                
                building_summary = building_summary.merge(suspicious_buildings, on='baglanti_nesnesi', how='left')
                building_summary['suheli_tesisat_sayisi'] = building_summary['suheli_tesisat_sayisi'].fillna(0)
                building_summary['suheli_oran'] = building_summary['suheli_tesisat_sayisi'] / building_summary['tesisat_sayisi'] * 100
            
            building_summary.to_excel(writer, sheet_name='Bina Bazında Analiz', index=False)
        
        # 5. Mevsimsel örüntü
        pattern_df = pd.DataFrame(list(seasonal_pattern.items()), columns=['Ay', 'Normalize_Tuketim'])
        pattern_df['Ay_Adi'] = pattern_df['Ay'].map({
            1: 'Ocak', 2: 'Şubat', 3: 'Mart', 4: 'Nisan', 5: 'Mayıs', 6: 'Haziran',
            7: 'Temmuz', 8: 'Ağustos', 9: 'Eylül', 10: 'Ekim', 11: 'Kasım', 12: 'Aralık'
        })
        pattern_df.to_excel(writer, sheet_name='Mevsimsel Örüntü', index=False)
    
    output.seek(0)
    return output

# Ana uygulama
if uploaded_file is not None:
    # Veriyi yükle
    with st.spinner("Veri yükleniyor ve işleniyor..."):
        df = load_and_process_data(uploaded_file)
    
    if df is not None:
        # Temel istatistikler
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Toplam Tesisat", df['tesisat_no'].nunique())
        
        with col2:
            st.metric("Toplam Bina", df['baglanti_nesnesi'].nunique())
        
        with col3:
            st.metric("Tarih Aralığı", f"{df['tarih'].min().strftime('%Y-%m')} - {df['tarih'].max().strftime('%Y-%m')}")
        
        with col4:
            st.metric("Toplam Kayıt", len(df))
        
        # Ek istatistikler
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.metric("Ortalama Tüketim", f"{df['sm3'].mean():.2f} sm³")
        
        with col6:
            winter_avg = df[df['is_winter']]['sm3'].mean()
            st.metric("Kış Ortalaması", f"{winter_avg:.2f} sm³")
        
        with col7:
            summer_avg = df[df['ay'].isin([6,7,8])]['sm3'].mean()
            st.metric("Yaz Ortalaması", f"{summer_avg:.2f} sm³")
        
        with col8:
            seasonal_diff = winter_avg - summer_avg
            st.metric("Mevsimsel Fark", f"{seasonal_diff:.2f} sm³")
        
        st.markdown("---")
        
        # Gelişmiş anomali tespiti
        with st.spinner("Gelişmiş anomali tespit algoritması çalışıyor..."):
            suspicious_df, seasonal_pattern, building_stats = advanced_anomaly_detection(
                df, winter_drop_threshold, building_avg_threshold, 
                pattern_deviation_threshold, min_months
            )
        
        # Sonuçlar
        if not suspicious_df.empty:
            # Risk kategorileri
            high_risk = len(suspicious_df[suspicious_df['risk_skoru'] > 200])
            medium_risk = len(suspicious_df[(suspicious_df['risk_skoru'] > 100) & (suspicious_df['risk_skoru'] <= 200)])
            low_risk = len(suspicious_df[suspicious_df['risk_skoru'] <= 100])
            
            st.success(f"🚨 {len(suspicious_df)} adet şüpheli tesisat tespit edildi!")
            
            # Risk dağılımı
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🔴 Yüksek Risk", high_risk, delta=f"{high_risk/len(suspicious_df)*100:.1f}%")
            with col2:
                st.metric("🟡 Orta Risk", medium_risk, delta=f"{medium_risk/len(suspicious_df)*100:.1f}%")
            with col3:
                st.metric("🟢 Düşük Risk", low_risk, delta=f"{low_risk/len(suspicious_df)*100:.1f}%")
            
            # Sonuçları göster
            st.subheader("🔍 Şüpheli Tesisatlar (Risk Skoruna Göre Sıralı)")
            
            # Risk seviyesine göre renklendirme
            def risk_color(risk_score):
                if risk_score > 200:
                    return 'background-color: #ffcdd2; color: #d32f2f'  # Kırmızı
                elif risk_score > 100:
                    return 'background-color: #ffe0b2; color: #f57c00'  # Turuncu
                else:
                    return 'background-color: #f3e5f5; color: #7b1fa2'  # Mor
            
            display_df = suspicious_df[['tesisat_no', 'baglanti_nesnesi', 'risk_skoru', 
                                      'yuksek_risk_anomali', 'orta_risk_anomali', 
                                      'dusuk_risk_anomali', 'anomali_tipleri', 
                                      'kis_ortalama', '
