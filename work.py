import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Doğalgaz Kaçak Kullanım Tespit Sistemi",
    page_icon="⚡",
    layout="wide"
)

st.title("🔍 Doğalgaz Kaçak Kullanım Tespit Sistemi")
st.markdown("---")

# Sidebar
st.sidebar.header("📊 Analiz Parametreleri")

# Veri yükleme fonksiyonu
@st.cache_data
def load_data():
    """Örnek veri oluşturma - gerçek verinizi buraya yükleyebilirsiniz"""
    # Örnek veri yapısı
    dates = pd.date_range(start='2018-01-01', end='2024-12-01', freq='MS')
    tesisatlar = [f"001000{i:03d}" for i in range(216, 550, 5)]  # Örnek tesisat numaraları
    
    data = []
    for tesisat in tesisatlar:
        for date in dates:
            # Normal tüketim paterni
            base_consumption = np.random.normal(400, 100)
            # Mevsimsel etki (kış aylarında daha fazla)
            seasonal_factor = 1.5 if date.month in [12, 1, 2] else 1.0
            # Bazı tesisatlarda kaçak kullanım simülasyonu
            if tesisat in tesisatlar[::10] and np.random.random() < 0.3:
                # Anormal düşük tüketim (kaçak kullanım)
                consumption = base_consumption * seasonal_factor * 0.3
            else:
                consumption = base_consumption * seasonal_factor
            
            data.append({
                'Tarih': date,
                'Tesisat_No': tesisat,
                'Baglanti_Nesnesi': f"BLK{tesisat[-3:]}",
                'SM3': max(0, consumption),
                'Yil': date.year,
                'Ay': date.month
            })
    
    return pd.DataFrame(data)

# Kaçak tespit algoritmaları
class KacakTespitAlgorithms:
    def __init__(self, df):
        self.df = df
        
    def istatistiksel_analiz(self, threshold_factor=2.5):
        """İstatistiksel anomali tespiti"""
        results = []
        
        for tesisat in self.df['Tesisat_No'].unique():
            tesisat_data = self.df[self.df['Tesisat_No'] == tesisat].copy()
            tesisat_data = tesisat_data.sort_values('Tarih')
            
            # Aylık tüketim ortalaması ve standart sapması
            mean_consumption = tesisat_data['SM3'].mean()
            std_consumption = tesisat_data['SM3'].std()
            
            # Anomali tespiti
            lower_bound = mean_consumption - threshold_factor * std_consumption
            upper_bound = mean_consumption + threshold_factor * std_consumption
            
            anomalies = tesisat_data[
                (tesisat_data['SM3'] < lower_bound) | 
                (tesisat_data['SM3'] > upper_bound)
            ]
            
            if len(anomalies) > 0:
                results.append({
                    'Tesisat_No': tesisat,
                    'Anomali_Sayisi': len(anomalies),
                    'Ortalama_Tuketim': mean_consumption,
                    'Min_Tuketim': tesisat_data['SM3'].min(),
                    'Max_Tuketim': tesisat_data['SM3'].max(),
                    'Risk_Skoru': len(anomalies) / len(tesisat_data) * 100
                })
        
        return pd.DataFrame(results)
    
    def mevsimsel_analiz(self):
        """Mevsimsel pattern analizi"""
        results = []
        
        for tesisat in self.df['Tesisat_No'].unique():
            tesisat_data = self.df[self.df['Tesisat_No'] == tesisat].copy()
            
            # Aylık ortalamalar
            monthly_avg = tesisat_data.groupby('Ay')['SM3'].mean()
            
            # Kış ayları (12, 1, 2) ile diğer aylar karşılaştırması
            winter_avg = monthly_avg[[12, 1, 2]].mean()
            other_avg = monthly_avg[[3, 4, 5, 6, 7, 8, 9, 10, 11]].mean()
            
            # Normal durumda kış tüketimi daha yüksek olmalı
            seasonal_ratio = winter_avg / other_avg if other_avg > 0 else 0
            
            # Anormal düşük mevsimsel oran kaçak kullanım işareti olabilir
            if seasonal_ratio < 1.2:  # Normal oran 1.5+ olmalı
                results.append({
                    'Tesisat_No': tesisat,
                    'Kis_Ortalama': winter_avg,
                    'Diger_Ortalama': other_avg,
                    'Mevsimsel_Oran': seasonal_ratio,
                    'Risk_Durumu': 'Yüksek Risk' if seasonal_ratio < 1.0 else 'Orta Risk'
                })
        
        return pd.DataFrame(results)
    
    def trend_analizi(self):
        """Tüketim trend analizi"""
        results = []
        
        for tesisat in self.df['Tesisat_No'].unique():
            tesisat_data = self.df[self.df['Tesisat_No'] == tesisat].copy()
            tesisat_data = tesisat_data.sort_values('Tarih')
            
            if len(tesisat_data) < 12:  # En az 1 yıl veri gerekli
                continue
            
            # Son 12 ay ve önceki 12 ay karşılaştırması
            recent_12 = tesisat_data.tail(12)['SM3'].mean()
            previous_12 = tesisat_data.iloc[-24:-12]['SM3'].mean() if len(tesisat_data) >= 24 else None
            
            if previous_12 and previous_12 > 0:
                change_rate = ((recent_12 - previous_12) / previous_12) * 100
                
                # Anormal düşüş kaçak kullanım işareti olabilir
                if change_rate < -30:  # %30'dan fazla düşüş
                    results.append({
                        'Tesisat_No': tesisat,
                        'Onceki_12_Ay': previous_12,
                        'Son_12_Ay': recent_12,
                        'Degisim_Orani': change_rate,
                        'Risk_Durumu': 'Yüksek Risk' if change_rate < -50 else 'Orta Risk'
                    })
        
        return pd.DataFrame(results)

# Ana uygulama
def main():
    # Veri yükleme
    df = load_data()
    
    # Sidebar parametreler
    st.sidebar.subheader("🔧 Tespit Parametreleri")
    threshold_factor = st.sidebar.slider("İstatistiksel Threshold", 1.0, 4.0, 2.5, 0.1)
    
    # Algoritma seçimi
    st.sidebar.subheader("🎯 Analiz Türü")
    analiz_turu = st.sidebar.selectbox(
        "Analiz türünü seçin:",
        ["Genel Bakış", "İstatistiksel Analiz", "Mevsimsel Analiz", "Trend Analizi", "Detaylı Tesisat Analizi"]
    )
    
    # Algoritma sınıfını başlat
    detector = KacakTespitAlgorithms(df)
    
    if analiz_turu == "Genel Bakış":
        st.header("📈 Genel Bakış")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Toplam Tesisat", len(df['Tesisat_No'].unique()))
        
        with col2:
            st.metric("Toplam Kayıt", len(df))
        
        with col3:
            avg_consumption = df['SM3'].mean()
            st.metric("Ortalama Tüketim", f"{avg_consumption:.2f} SM3")
        
        with col4:
            total_consumption = df['SM3'].sum()
            st.metric("Toplam Tüketim", f"{total_consumption:,.0f} SM3")
        
        # Aylık toplam tüketim grafiği
        monthly_total = df.groupby('Tarih')['SM3'].sum().reset_index()
        fig = px.line(monthly_total, x='Tarih', y='SM3', 
                     title="Aylık Toplam Doğalgaz Tüketimi")
        st.plotly_chart(fig, use_container_width=True)
        
        # Tüketim dağılımı
        fig = px.histogram(df, x='SM3', nbins=50, 
                          title="Tüketim Dağılımı")
        st.plotly_chart(fig, use_container_width=True)
    
    elif analiz_turu == "İstatistiksel Analiz":
        st.header("📊 İstatistiksel Anomali Analizi")
        
        anomalies_df = detector.istatistiksel_analiz(threshold_factor)
        
        if not anomalies_df.empty:
            st.subheader("🚨 Şüpheli Tesisatlar")
            
            # Risk skoruna göre sıralama
            anomalies_df = anomalies_df.sort_values('Risk_Skoru', ascending=False)
            
            # Renkli tablo
            st.dataframe(
                anomalies_df.style.background_gradient(subset=['Risk_Skoru'], cmap='Reds'),
                use_container_width=True
            )
            
            # Risk skoru dağılımı
            fig = px.bar(anomalies_df.head(20), x='Tesisat_No', y='Risk_Skoru',
                        title="En Yüksek Risk Skorlu Tesisatlar")
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Belirlenen parametrelere göre anomali tespit edilmedi.")
    
    elif analiz_turu == "Mevsimsel Analiz":
        st.header("❄️ Mevsimsel Pattern Analizi")
        
        seasonal_df = detector.mevsimsel_analiz()
        
        if not seasonal_df.empty:
            st.subheader("🔍 Mevsimsel Anomaliler")
            
            # Risk durumuna göre renklendirme
            def risk_color(val):
                if val == 'Yüksek Risk':
                    return 'background-color: #ffcccc'
                elif val == 'Orta Risk':
                    return 'background-color: #fff3cd'
                return ''
            
            st.dataframe(
                seasonal_df.style.applymap(risk_color, subset=['Risk_Durumu']),
                use_container_width=True
            )
            
            # Mevsimsel oran dağılımı
            fig = px.histogram(seasonal_df, x='Mevsimsel_Oran', 
                             title="Mevsimsel Oran Dağılımı")
            fig.add_vline(x=1.2, line_dash="dash", line_color="red", 
                         annotation_text="Normal Threshold")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Mevsimsel anomali tespit edilmedi.")
    
    elif analiz_turu == "Trend Analizi":
        st.header("📈 Tüketim Trend Analizi")
        
        trend_df = detector.trend_analizi()
        
        if not trend_df.empty:
            st.subheader("📉 Anormal Tüketim Değişiklikleri")
            
            def trend_color(val):
                if val == 'Yüksek Risk':
                    return 'background-color: #ffcccc'
                elif val == 'Orta Risk':
                    return 'background-color: #fff3cd'
                return ''
            
            st.dataframe(
                trend_df.style.applymap(trend_color, subset=['Risk_Durumu']),
                use_container_width=True
            )
            
            # Değişim oranı dağılımı
            fig = px.bar(trend_df, x='Tesisat_No', y='Degisim_Orani',
                        title="Tüketim Değişim Oranları (%)")
            fig.add_hline(y=-30, line_dash="dash", line_color="red",
                         annotation_text="Risk Threshold")
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Anormal trend tespit edilmedi.")
    
    elif analiz_turu == "Detaylı Tesisat Analizi":
        st.header("🔍 Detaylı Tesisat Analizi")
        
        # Tesisat seçimi
        selected_tesisat = st.selectbox(
            "Analiz edilecek tesisatı seçin:",
            df['Tesisat_No'].unique()
        )
        
        if selected_tesisat:
            tesisat_data = df[df['Tesisat_No'] == selected_tesisat].copy()
            tesisat_data = tesisat_data.sort_values('Tarih')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Ortalama Tüketim", f"{tesisat_data['SM3'].mean():.2f} SM3")
                st.metric("Minimum Tüketim", f"{tesisat_data['SM3'].min():.2f} SM3")
            
            with col2:
                st.metric("Maksimum Tüketim", f"{tesisat_data['SM3'].max():.2f} SM3")
                st.metric("Standart Sapma", f"{tesisat_data['SM3'].std():.2f} SM3")
            
            # Zaman serisi grafiği
            fig = px.line(tesisat_data, x='Tarih', y='SM3',
                         title=f"Tesisat {selected_tesisat} - Zaman Serisi")
            
            # Ortalama çizgisi
            mean_val = tesisat_data['SM3'].mean()
            fig.add_hline(y=mean_val, line_dash="dash", line_color="green",
                         annotation_text=f"Ortalama: {mean_val:.2f}")
            
            # Anomali thresholdları
            std_val = tesisat_data['SM3'].std()
            fig.add_hline(y=mean_val + 2*std_val, line_dash="dash", line_color="red",
                         annotation_text="Üst Threshold")
            fig.add_hline(y=mean_val - 2*std_val, line_dash="dash", line_color="red",
                         annotation_text="Alt Threshold")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Aylık dağılım
            monthly_consumption = tesisat_data.groupby('Ay')['SM3'].mean().reset_index()
            monthly_consumption['Ay_Adi'] = monthly_consumption['Ay'].map({
                1: 'Ocak', 2: 'Şubat', 3: 'Mart', 4: 'Nisan', 5: 'Mayıs', 6: 'Haziran',
                7: 'Temmuz', 8: 'Ağustos', 9: 'Eylül', 10: 'Ekim', 11: 'Kasım', 12: 'Aralık'
            })
            
            fig = px.bar(monthly_consumption, x='Ay_Adi', y='SM3',
                        title="Aylık Ortalama Tüketim")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
