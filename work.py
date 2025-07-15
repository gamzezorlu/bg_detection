import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
from scipy import stats
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
seasonal_threshold = st.sidebar.slider(
    "Mevsimsel Sapma Eşiği (%)",
    min_value=10,
    max_value=80,
    value=40,
    help="Mevsimsel ortalamadan sapma yüzdesi"
)

trend_threshold = st.sidebar.slider(
    "Trend Değişim Eşiği (%)",
    min_value=20,
    max_value=90,
    value=50,
    help="Ani düşüş/artış tespit eşiği"
)

min_months = st.sidebar.slider(
    "Minimum Veri Süresi (Ay)",
    min_value=6,
    max_value=24,
    value=12,
    help="Analiz için minimum veri süresi"
)

def load_and_process_data(file):
    """Veriyi yükle ve işle"""
    try:
        # Excel dosyasını oku
        df = pd.read_excel(file)
        
        # Sütun adlarını standartlaştır
        df.columns = df.columns.str.lower().str.strip()
        
        # Gerekli sütunları kontrol et
        required_cols = ['tesisat_no', 'tarih', 'baglanti_nesnesi', 'sm3']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Eksik sütunlar: {missing_cols}")
            return None
        
        # Tarih sütununu datetime'a çevir
        df['tarih'] = pd.to_datetime(df['tarih'])
        
        # Numerik sütunları kontrol et
        df['sm3'] = pd.to_numeric(df['sm3'], errors='coerce')
        df['tesisat_no'] = pd.to_numeric(df['tesisat_no'], errors='coerce')
        
        # Eksik verileri temizle
        df = df.dropna(subset=['tesisat_no', 'tarih', 'sm3'])
        
        # Tarih indeksi oluştur
        df['yil'] = df['tarih'].dt.year
        df['ay'] = df['tarih'].dt.month
        df['mevsim'] = df['ay'].apply(lambda x: 'Kış' if x in [12, 1, 2] else 
                                               'İlkbahar' if x in [3, 4, 5] else 
                                               'Yaz' if x in [6, 7, 8] else 'Sonbahar')
        
        return df
    
    except Exception as e:
        st.error(f"Veri yükleme hatası: {str(e)}")
        return None

def detect_anomalies(df, seasonal_threshold, trend_threshold, min_months):
    """Kaçak kullanım tespiti"""
    suspicious_facilities = []
    
    # Her tesisat için analiz
    for tesisat in df['tesisat_no'].unique():
        facility_data = df[df['tesisat_no'] == tesisat].copy()
        facility_data = facility_data.sort_values('tarih')
        
        # Minimum veri kontrolü
        if len(facility_data) < min_months:
            continue
        
        # Mevsimsel ortalamalar
        seasonal_avg = facility_data.groupby('mevsim')['sm3'].mean()
        
        # Trend analizi
        facility_data['rolling_avg'] = facility_data['sm3'].rolling(window=3, min_periods=1).mean()
        
        # Anomali tespiti
        anomalies = []
        
        # 1. Mevsimsel anormallik
        for _, row in facility_data.iterrows():
            season_avg = seasonal_avg.get(row['mevsim'], facility_data['sm3'].mean())
            if season_avg > 0:
                deviation = abs(row['sm3'] - season_avg) / season_avg * 100
                if deviation > seasonal_threshold:
                    anomalies.append({
                        'type': 'Mevsimsel Anormallik',
                        'date': row['tarih'],
                        'value': row['sm3'],
                        'expected': season_avg,
                        'deviation': deviation
                    })
        
        # 2. Ani düşüş/artış
        for i in range(1, len(facility_data)):
            current = facility_data.iloc[i]['sm3']
            previous = facility_data.iloc[i-1]['sm3']
            
            if previous > 0:
                change = abs(current - previous) / previous * 100
                if change > trend_threshold:
                    anomalies.append({
                        'type': 'Ani Değişim',
                        'date': facility_data.iloc[i]['tarih'],
                        'value': current,
                        'expected': previous,
                        'deviation': change
                    })
        
        # 3. Sürekli düşük tüketim (yaz ayları hariç)
        winter_spring_data = facility_data[facility_data['mevsim'].isin(['Kış', 'İlkbahar'])]
        if len(winter_spring_data) > 0:
            avg_consumption = winter_spring_data['sm3'].mean()
            low_consumption_months = winter_spring_data[winter_spring_data['sm3'] < avg_consumption * 0.3]
            
            if len(low_consumption_months) > 2:
                anomalies.append({
                    'type': 'Sürekli Düşük Tüketim',
                    'date': low_consumption_months.iloc[0]['tarih'],
                    'value': low_consumption_months['sm3'].mean(),
                    'expected': avg_consumption,
                    'deviation': 70
                })
        
        # 4. Sıfır tüketim
        zero_consumption = facility_data[facility_data['sm3'] == 0]
        if len(zero_consumption) > 1:
            anomalies.append({
                'type': 'Sıfır Tüketim',
                'date': zero_consumption.iloc[0]['tarih'],
                'value': 0,
                'expected': facility_data['sm3'].mean(),
                'deviation': 100
            })
        
        # Şüpheli tesisatları kaydet
        if anomalies:
            risk_score = len(anomalies) + sum([a['deviation'] for a in anomalies]) / len(anomalies)
            
            suspicious_facilities.append({
                'tesisat_no': tesisat,
                'baglanti_nesnesi': facility_data['baglanti_nesnesi'].iloc[0],
                'toplam_anomali': len(anomalies),
                'risk_skoru': risk_score,
                'anomali_tipleri': ', '.join(list(set([a['type'] for a in anomalies]))),
                'ortalama_tuketim': facility_data['sm3'].mean(),
                'son_tuketim': facility_data['sm3'].iloc[-1],
                'ilk_anomali_tarihi': min([a['date'] for a in anomalies]),
                'anomali_detaylari': anomalies
            })
    
    return pd.DataFrame(suspicious_facilities)

def create_visualizations(df, suspicious_df):
    """Görselleştirmeler oluştur"""
    
    # 1. Genel tüketim trendi
    monthly_consumption = df.groupby(['yil', 'ay'])['sm3'].sum().reset_index()
    monthly_consumption['tarih'] = pd.to_datetime(monthly_consumption[['yil', 'ay']].assign(day=1))
    
    fig1 = px.line(
        monthly_consumption, 
        x='tarih', 
        y='sm3',
        title='Toplam Aylık Doğalgaz Tüketimi',
        labels={'sm3': 'Tüketim (sm³)', 'tarih': 'Tarih'}
    )
    fig1.update_layout(height=400)
    
    # 2. Mevsimsel analiz
    seasonal_data = df.groupby('mevsim')['sm3'].mean().reset_index()
    fig2 = px.bar(
        seasonal_data,
        x='mevsim',
        y='sm3',
        title='Mevsimsel Ortalama Tüketim',
        labels={'sm3': 'Ortalama Tüketim (sm³)', 'mevsim': 'Mevsim'}
    )
    fig2.update_layout(height=400)
    
    # 3. Risk skoru dağılımı
    if not suspicious_df.empty:
        fig3 = px.histogram(
            suspicious_df,
            x='risk_skoru',
            nbins=20,
            title='Şüpheli Tesisatlar Risk Skoru Dağılımı',
            labels={'risk_skoru': 'Risk Skoru', 'count': 'Tesisat Sayısı'}
        )
        fig3.update_layout(height=400)
    else:
        fig3 = go.Figure()
        fig3.add_annotation(text="Şüpheli tesisat bulunamadı", x=0.5, y=0.5)
        fig3.update_layout(height=400, title="Risk Skoru Dağılımı")
    
    return fig1, fig2, fig3

def export_results(suspicious_df, df):
    """Sonuçları Excel formatında hazırla"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Şüpheli tesisatlar
        export_df = suspicious_df.copy()
        if 'anomali_detaylari' in export_df.columns:
            export_df = export_df.drop('anomali_detaylari', axis=1)
        
        export_df.to_excel(writer, sheet_name='Şüpheli Tesisatlar', index=False)
        
        # Genel istatistikler
        stats_df = pd.DataFrame({
            'Metrik': [
                'Toplam Tesisat Sayısı',
                'Şüpheli Tesisat Sayısı',
                'Şüpheli Oran (%)',
                'Ortalama Risk Skoru',
                'Yüksek Risk Tesisat (>100)',
                'Analiz Edilen Dönem'
            ],
            'Değer': [
                df['tesisat_no'].nunique(),
                len(suspicious_df),
                round(len(suspicious_df) / df['tesisat_no'].nunique() * 100, 2) if df['tesisat_no'].nunique() > 0 else 0,
                round(suspicious_df['risk_skoru'].mean(), 2) if not suspicious_df.empty else 0,
                len(suspicious_df[suspicious_df['risk_skoru'] > 100]) if not suspicious_df.empty else 0,
                f"{df['tarih'].min().strftime('%Y-%m')} - {df['tarih'].max().strftime('%Y-%m')}"
            ]
        })
        
        stats_df.to_excel(writer, sheet_name='Genel İstatistikler', index=False)
    
    output.seek(0)
    return output

# Ana uygulama
if uploaded_file is not None:
    # Veriyi yükle
    with st.spinner("Veri yükleniyor..."):
        df = load_and_process_data(uploaded_file)
    
    if df is not None:
        # Temel istatistikler
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Toplam Tesisat", df['tesisat_no'].nunique())
        
        with col2:
            st.metric("Toplam Kayıt", len(df))
        
        with col3:
            st.metric("Tarih Aralığı", f"{df['tarih'].min().strftime('%Y-%m')} - {df['tarih'].max().strftime('%Y-%m')}")
        
        with col4:
            st.metric("Ortalama Tüketim", f"{df['sm3'].mean():.2f} sm³")
        
        st.markdown("---")
        
        # Anomali tespiti
        with st.spinner("Kaçak kullanım analizi yapılıyor..."):
            suspicious_df = detect_anomalies(df, seasonal_threshold, trend_threshold, min_months)
        
        # Sonuçlar
        if not suspicious_df.empty:
            st.success(f"🚨 {len(suspicious_df)} adet şüpheli tesisat tespit edildi!")
            
            # Sonuçları göster
            st.subheader("🔍 Şüpheli Tesisatlar")
            
            # Risk seviyesine göre renklendirme
            def risk_color(risk_score):
                if risk_score > 150:
                    return 'background-color: #ffebee'  # Kırmızı
                elif risk_score > 100:
                    return 'background-color: #fff3e0'  # Turuncu
                else:
                    return 'background-color: #f3e5f5'  # Mor
            
            display_df = suspicious_df[['tesisat_no', 'baglanti_nesnesi', 'toplam_anomali', 
                                      'risk_skoru', 'anomali_tipleri', 'ortalama_tuketim', 
                                      'son_tuketim', 'ilk_anomali_tarihi']].copy()
            
            display_df['risk_skoru'] = display_df['risk_skoru'].round(2)
            display_df['ortalama_tuketim'] = display_df['ortalama_tuketim'].round(2)
            display_df['ilk_anomali_tarihi'] = display_df['ilk_anomali_tarihi'].dt.strftime('%Y-%m-%d')
            
            # Sıralama
            display_df = display_df.sort_values('risk_skoru', ascending=False)
            
            st.dataframe(
                display_df.style.apply(lambda x: [risk_color(val) if col == 'risk_skoru' else '' 
                                                for col, val in x.items()], axis=1),
                use_container_width=True
            )
            
            # Görselleştirmeler
            st.subheader("📊 Analiz Grafikleri")
            
            fig1, fig2, fig3 = create_visualizations(df, suspicious_df)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                st.plotly_chart(fig2, use_container_width=True)
            
            st.plotly_chart(fig3, use_container_width=True)
            
            # Detaylı analiz
            st.subheader("🔍 Detaylı Tesisat Analizi")
            selected_facility = st.selectbox(
                "Analiz edilecek tesisatı seçin:",
                suspicious_df['tesisat_no'].tolist()
            )
            
            if selected_facility:
                facility_data = df[df['tesisat_no'] == selected_facility].copy()
                facility_data = facility_data.sort_values('tarih')
                
                # Tesisat grafiği
                fig_facility = px.line(
                    facility_data,
                    x='tarih',
                    y='sm3',
                    title=f'Tesisat {selected_facility} - Tüketim Trendi',
                    labels={'sm3': 'Tüketim (sm³)', 'tarih': 'Tarih'}
                )
                
                # Mevsimsel ortalama çizgisi ekle
                seasonal_avg = facility_data.groupby('mevsim')['sm3'].mean()
                for season, avg in seasonal_avg.items():
                    season_data = facility_data[facility_data['mevsim'] == season]
                    if not season_data.empty:
                        fig_facility.add_hline(
                            y=avg,
                            line_dash="dash",
                            annotation_text=f"{season} Ort: {avg:.2f}",
                            annotation_position="top left"
                        )
                
                st.plotly_chart(fig_facility, use_container_width=True)
                
                # Anomali detayları
                facility_anomalies = suspicious_df[suspicious_df['tesisat_no'] == selected_facility]['anomali_detaylari'].iloc[0]
                
                st.subheader("⚠️ Tespit Edilen Anomaliler")
                for i, anomaly in enumerate(facility_anomalies):
                    st.write(f"**{i+1}. {anomaly['type']}**")
                    st.write(f"- Tarih: {anomaly['date'].strftime('%Y-%m-%d')}")
                    st.write(f"- Ölçülen Değer: {anomaly['value']:.2f} sm³")
                    st.write(f"- Beklenen Değer: {anomaly['expected']:.2f} sm³")
                    st.write(f"- Sapma: %{anomaly['deviation']:.2f}")
                    st.write("---")
            
            # Excel export
            st.subheader("📥 Sonuçları İndir")
            excel_file = export_results(suspicious_df, df)
            
            st.download_button(
                label="📊 Excel Raporu İndir",
                data=excel_file,
                file_name=f"kacak_kullanim_raporu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        else:
            st.info("✅ Mevcut parametrelerle şüpheli tesisat tespit edilmedi.")
            st.write("Analiz parametrelerini düşürerek tekrar deneyin.")
    
else:
    st.info("📁 Lütfen analiz edilecek Excel dosyasını yükleyin.")
    
    # Örnek veri formatı
    st.subheader("📋 Beklenen Veri Formatı")
    sample_data = pd.DataFrame({
        'tesisat_no': [1001, 1001, 1002, 1002],
        'tarih': ['2024-01-01', '2024-02-01', '2024-01-01', '2024-02-01'],
        'baglanti_nesnesi': [100003156, 100003156, 100003157, 100003157],
        'sm3': [500.13, 450.25, 380.75, 420.50]
    })
    
    st.dataframe(sample_data, use_container_width=True)
    
    st.markdown("""
    **Sütun Açıklamaları:**
    - **tesisat_no**: Tesisat numarası
    - **tarih**: Ölçüm tarihi (YYYY-MM-DD formatında)
    - **baglanti_nesnesi**: Tesisatın bağlı olduğu bina numarası
    - **sm3**: Aylık doğalgaz tüketimi (standart metreküp)
    """)

# Footer
st.markdown("---")
st.markdown("🔍 **Doğalgaz Kaçak Kullanım Tespit Sistemi** - Gelişmiş analiz ve raporlama özellikleri")
