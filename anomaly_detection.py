import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Doğalgaz Tüketim Anomali Tespit",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 Doğalgaz Tüketim Anomali Tespit Sistemi")
st.markdown("---")

# Yan panel - Dosya yükleme
st.sidebar.header("📁 Dosya Yükleme")
uploaded_file = st.sidebar.file_uploader(
    "CSV veya Excel dosyası seçin",
    type=['csv', 'xlsx', 'xls'],
    help="Tesisat numarası, bina numarası ve aylık tüketim verilerini içeren dosya"
)

# Parametreler
st.sidebar.header("⚙️ Analiz Parametreleri")
kis_tuketim_esigi = st.sidebar.slider(
    "Kış ayı düşük tüketim eşiği (m³/ay)",
    min_value=10, max_value=100, value=30,
    help="Kış aylarında bu değerin altındaki tüketim şüpheli kabul edilir"
)

bina_ort_dusuk_oran = st.sidebar.slider(
    "Bina ortalamasından düşük olma oranı (%)",
    min_value=30, max_value=90, value=60,
    help="Bina ortalamasından bu oranda düşük tüketim şüpheli kabul edilir"
)

ani_dusus_orani = st.sidebar.slider(
    "Ani düşüş oranı (%)",
    min_value=40, max_value=90, value=70,
    help="Önceki kış aylarına göre bu oranda düşüş şüpheli kabul edilir"
)

min_onceki_kis_tuketim = st.sidebar.slider(
    "Minimum önceki kış tüketimi (m³)",
    min_value=50, max_value=200, value=100,
    help="Ani düşüş tespiti için önceki kış aylarında minimum tüketim"
)

def load_data(file):
    """Dosyayı yükle ve temizle"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        # Kolon isimlerini temizle
        df.columns = df.columns.str.strip()
        
        return df
    except Exception as e:
        st.error(f"Dosya yükleme hatası: {str(e)}")
        return None

def parse_date_columns(df):
    """Tarih sütunlarını parse et"""
    date_columns = []
    other_columns = []
    
    for col in df.columns:
        if isinstance(col, str) and '/' in col:
            try:
                year, month = col.split('/')
                if len(year) == 4 and len(month) <= 2:
                    date_columns.append(col)
                else:
                    other_columns.append(col)
            except:
                other_columns.append(col)
        else:
            other_columns.append(col)
    
    return date_columns, other_columns

def get_season(month):
    """Ayı mevsime göre kategorize et"""
    if month in [12, 1, 2]:
        return "Kış"
    elif month in [3, 4, 5]:
        return "İlkbahar"
    elif month in [6, 7, 8]:
        return "Yaz"
    else:
        return "Sonbahar"

def analyze_consumption_patterns(df, date_columns, tesisat_col, bina_col):
    """Tüketim paternlerini analiz et"""
    results = []
    
    for idx, row in df.iterrows():
        tesisat_no = row[tesisat_col]
        bina_no = row[bina_col]
        
        # Aylık tüketim verilerini al
        consumption_data = []
        for date_col in date_columns:
            try:
                value = row[date_col]
                if pd.notna(value):
                    year, month = date_col.split('/')
                    consumption_data.append({
                        'year': int(year),
                        'month': int(month),
                        'consumption': float(value) if value != 0 else 0,
                        'season': get_season(int(month)),
                        'date_str': date_col
                    })
            except:
                continue
        
        if not consumption_data:
            continue
        
        # DataFrame'e çevir ve tarihine göre sırala
        cons_df = pd.DataFrame(consumption_data)
        cons_df = cons_df.sort_values(['year', 'month'])
        
        # Mevsimsel ortalamalar (sıfır olmayan değerler için)
        seasonal_avg = cons_df[cons_df['consumption'] > 0].groupby('season')['consumption'].mean()
        
        # Kış ayı tüketimi kontrolü
        kis_tuketim = seasonal_avg.get('Kış', 0)
        yaz_tuketim = seasonal_avg.get('Yaz', 0)
        
        # Anomali tespiti
        anomalies = []
        
        # 1. Kış ayı düşük tüketim
        if kis_tuketim < kis_tuketim_esigi and kis_tuketim > 0:
            anomalies.append(f"Kış ayı düşük tüketim: {kis_tuketim:.1f} m³/ay")
        
        # 2. Kış-yaz tüketim farkı normal değil
        if kis_tuketim > 0 and yaz_tuketim > 0:
            if abs(kis_tuketim - yaz_tuketim) < 10:  # Fark çok az
                anomalies.append(f"Kış-yaz tüketim farkı az: Kış {kis_tuketim:.1f}, Yaz {yaz_tuketim:.1f}")
        
        # 3. Toplam tüketim çok düşük
        total_consumption = cons_df['consumption'].sum()
        if total_consumption < 100:  # Yıllık 100 m³'den az
            anomalies.append(f"Toplam tüketim çok düşük: {total_consumption:.1f} m³")
        
        # 4. Düzenli sıfır tüketim
        zero_months = len(cons_df[cons_df['consumption'] == 0])
        if zero_months > 6:
            anomalies.append(f"Çok fazla sıfır tüketim: {zero_months} ay")
        
        # 5. ANI DÜŞÜŞ TESPİTİ - Kış aylarında ani düşüş
        kis_aylari = cons_df[cons_df['season'] == 'Kış'].copy()
        if len(kis_aylari) >= 4:  # En az 2 kış sezonu olmalı
            # Yıllara göre kış aylarını grupla
            kis_yillik = kis_aylari.groupby('year')['consumption'].mean()
            
            # Yıllık kış ortalamaları al (sıfır olmayan)
            yillik_ortalamalar = kis_yillik[kis_yillik > 0]
            
            if len(yillik_ortalamalar) >= 2:
                # En az 2 yıl veri varsa ani düşüş kontrolü yap
                yillar = sorted(yillik_ortalamalar.index)
                
                for i in range(1, len(yillar)):
                    onceki_yil = yillar[i-1]
                    mevcut_yil = yillar[i]
                    
                    onceki_tuketim = yillik_ortalamalar[onceki_yil]
                    mevcut_tuketim = yillik_ortalamalar[mevcut_yil]
                    
                    # Önceki kış yüksek tüketim ve ani düşüş kontrolü
                    if (onceki_tuketim >= min_onceki_kis_tuketim and 
                        mevcut_tuketim < onceki_tuketim * (1 - ani_dusus_orani/100)):
                        
                        dusus_orani = ((onceki_tuketim - mevcut_tuketim) / onceki_tuketim) * 100
                        anomalies.append(f"Ani kış düşüşü: {onceki_yil} ({onceki_tuketim:.1f}) → {mevcut_yil} ({mevcut_tuketim:.1f}), %{dusus_orani:.1f} düşüş")
                
                # Son 2 yıl özel kontrolü
                if len(yillar) >= 2:
                    son_iki_yil = yillar[-2:]
                    if len(son_iki_yil) == 2:
                        onceki_son = yillik_ortalamalar[son_iki_yil[0]]
                        mevcut_son = yillik_ortalamalar[son_iki_yil[1]]
                        
                        if (onceki_son >= min_onceki_kis_tuketim and 
                            mevcut_son < onceki_son * (1 - ani_dusus_orani/100)):
                            
                            dusus_orani = ((onceki_son - mevcut_son) / onceki_son) * 100
                            anomalies.append(f"Son yıl ani düşüş: {son_iki_yil[0]} → {son_iki_yil[1]}, %{dusus_orani:.1f} düşüş")
        
        # 6. Bina ortalaması kontrolü (aynı binadaki diğer tesisatlarla karşılaştır)
        bina_tesisatlari = df[df[bina_col] == bina_no]
        if len(bina_tesisatlari) > 1:
            bina_tuketimleri = []
            for _, bina_row in bina_tesisatlari.iterrows():
                bina_toplam = 0
                bina_ay_sayisi = 0
                for date_col in date_columns:
                    try:
                        val = bina_row[date_col]
                        if pd.notna(val) and val > 0:
                            bina_toplam += float(val)
                            bina_ay_sayisi += 1
                    except:
                        continue
                
                if bina_ay_sayisi > 0:
                    bina_tuketimleri.append(bina_toplam / bina_ay_sayisi)
            
            if len(bina_tuketimleri) > 1:
                bina_ortalaması = np.mean(bina_tuketimleri)
                mevcut_ortalama = cons_df[cons_df['consumption'] > 0]['consumption'].mean() if len(cons_df[cons_df['consumption'] > 0]) > 0 else 0
                
                if mevcut_ortalama > 0 and mevcut_ortalama < bina_ortalaması * (1 - bina_ort_dusuk_oran/100):
                    anomalies.append(f"Bina ortalamasından %{bina_ort_dusuk_oran} düşük: {mevcut_ortalama:.1f} vs {bina_ortalaması:.1f}")
        
        # Ani düşüş bilgisi için ek analiz
        kis_trend = "Stabil"
        if len(kis_aylari) >= 4:
            kis_yillik = kis_aylari.groupby('year')['consumption'].mean()
            yillik_ortalamalar = kis_yillik[kis_yillik > 0]
            
            if len(yillik_ortalamalar) >= 2:
                yillar = sorted(yillik_ortalamalar.index)
                ilk_yil = yillik_ortalamalar[yillar[0]]
                son_yil = yillik_ortalamalar[yillar[-1]]
                
                if son_yil < ilk_yil * 0.5:
                    kis_trend = "Şiddetli Düşüş"
                elif son_yil < ilk_yil * 0.7:
                    kis_trend = "Orta Düşüş"
                elif son_yil > ilk_yil * 1.5:
                    kis_trend = "Artış"
        
        # Sonuçları kaydet
        results.append({
            'tesisat_no': tesisat_no,
            'bina_no': bina_no,
            'kis_tuketim': kis_tuketim,
            'yaz_tuketim': yaz_tuketim,
            'toplam_tuketim': total_consumption,
            'ortalama_tuketim': cons_df[cons_df['consumption'] > 0]['consumption'].mean() if len(cons_df[cons_df['consumption'] > 0]) > 0 else 0,
            'kis_trend': kis_trend,
            'anomali_sayisi': len(anomalies),
            'anomaliler': '; '.join(anomalies) if anomalies else 'Normal',
            'suspicion_level': 'Şüpheli' if anomalies else 'Normal'
        })
    
    return pd.DataFrame(results)

def create_visualizations(results_df, original_df, date_columns):
    """Görselleştirmeler oluştur"""
    
    # 1. Anomali dağılımı
    fig1 = px.histogram(
        results_df, 
        x='anomali_sayisi',
        title="Anomali Sayısı Dağılımı",
        color_discrete_sequence=['#FF6B6B']
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # 2. Şüpheli vs Normal dağılımı
    suspicion_counts = results_df['suspicion_level'].value_counts()
    fig2 = px.pie(
        values=suspicion_counts.values,
        names=suspicion_counts.index,
        title="Şüpheli vs Normal Tesisatlar",
        color_discrete_map={'Şüpheli': '#FF6B6B', 'Normal': '#4ECDC4'}
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # 3. Kış Trend Analizi
    trend_counts = results_df['kis_trend'].value_counts()
    fig3 = px.bar(
        x=trend_counts.index,
        y=trend_counts.values,
        title="Kış Ayı Tüketim Trend Analizi",
        color=trend_counts.values,
        color_continuous_scale='Reds'
    )
    fig3.update_layout(showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)
    
    # 4. Kış vs Yaz tüketim karşılaştırması
    fig4 = px.scatter(
        results_df,
        x='yaz_tuketim',
        y='kis_tuketim',
        color='suspicion_level',
        size='anomali_sayisi',
        title="Kış vs Yaz Tüketim Karşılaştırması",
        labels={'yaz_tuketim': 'Yaz Tüketimi (m³)', 'kis_tuketim': 'Kış Tüketimi (m³)'},
        color_discrete_map={'Şüpheli': '#FF6B6B', 'Normal': '#4ECDC4'},
        hover_data=['kis_trend']
    )
    
    # Normal pattern çizgisi ekle
    max_val = max(results_df['yaz_tuketim'].max(), results_df['kis_tuketim'].max())
    fig4.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        name='Eşit Tüketim Çizgisi',
        line=dict(dash='dash', color='gray')
    ))
    
    st.plotly_chart(fig4, use_container_width=True)
    
    # 5. Trend bazında anomali dağılımı
    trend_anomali = results_df.groupby(['kis_trend', 'suspicion_level']).size().reset_index(name='count')
    fig5 = px.bar(
        trend_anomali,
        x='kis_trend',
        y='count',
        color='suspicion_level',
        title="Trend Bazında Anomali Dağılımı",
        color_discrete_map={'Şüpheli': '#FF6B6B', 'Normal': '#4ECDC4'}
    )
    st.plotly_chart(fig5, use_container_width=True)

# Ana uygulama
if uploaded_file is not None:
    # Veri yükleme
    df = load_data(uploaded_file)
    
    if df is not None:
        st.success("✅ Dosya başarıyla yüklendi!")
        
        # Veri önizleme
        st.subheader("📊 Veri Önizleme")
        st.dataframe(df.head())
        
        # Sütun seçimi
        st.subheader("🔧 Sütun Seçimi")
        
        date_columns, other_columns = parse_date_columns(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            tesisat_col = st.selectbox(
                "Tesisat Numarası Sütunu",
                options=other_columns,
                help="Tesisat numarasını içeren sütunu seçin"
            )
        
        with col2:
            bina_col = st.selectbox(
                "Bina Numarası Sütunu",
                options=other_columns,
                help="Bina numarasını içeren sütunu seçin"
            )
        
        # Tarih sütunlarını göster
        st.write(f"**Tespit edilen tarih sütunları:** {len(date_columns)} adet")
        st.write(f"Tarih aralığı: {min(date_columns)} - {max(date_columns)}")
        
        # Analiz butonu
        if st.button("🔍 Anomali Analizini Başlat", type="primary"):
            with st.spinner("Analiz yapılıyor..."):
                
                # Analiz yap
                results_df = analyze_consumption_patterns(df, date_columns, tesisat_col, bina_col)
                
                # Sonuçları göster
                st.subheader("📈 Analiz Sonuçları")
                
                # Özet istatistikler
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Toplam Tesisat", len(results_df))
                
                with col2:
                    suspicious_count = len(results_df[results_df['suspicion_level'] == 'Şüpheli'])
                    st.metric("Şüpheli Tesisat", suspicious_count)
                
                with col3:
                    if len(results_df) > 0:
                        suspicious_rate = (suspicious_count / len(results_df)) * 100
                        st.metric("Şüpheli Oran", f"{suspicious_rate:.1f}%")
                
                with col4:
                    total_anomalies = results_df['anomali_sayisi'].sum()
                    st.metric("Toplam Anomali", total_anomalies)
                
                # Görselleştirmeler
                st.subheader("📊 Görselleştirmeler")
                create_visualizations(results_df, df, date_columns)
                
                # Şüpheli tesisatlar tablosu
                st.subheader("🚨 Şüpheli Tesisatlar")
                suspicious_df = results_df[results_df['suspicion_level'] == 'Şüpheli'].copy()
                
                if not suspicious_df.empty:
                    # Sütunları düzenle
                    display_cols = ['tesisat_no', 'bina_no', 'kis_tuketim', 'yaz_tuketim', 
                                   'ortalama_tuketim', 'kis_trend', 'anomali_sayisi', 'anomaliler']
                    
                    suspicious_display = suspicious_df[display_cols].copy()
                    suspicious_display.columns = ['Tesisat No', 'Bina No', 'Kış Tüketim', 
                                                'Yaz Tüketim', 'Ortalama Tüketim', 'Kış Trend',
                                                'Anomali Sayısı', 'Anomaliler']
                    
                    # Numeric columns için formatting
                    for col in ['Kış Tüketim', 'Yaz Tüketim', 'Ortalama Tüketim']:
                        suspicious_display[col] = suspicious_display[col].round(1)
                    
                    st.dataframe(
                        suspicious_display,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Excel indirme
                    csv = suspicious_display.to_csv(index=False)
                    st.download_button(
                        label="📥 Şüpheli Tesisatları İndir (CSV)",
                        data=csv,
                        file_name="supheli_tesisatlar.csv",
                        mime="text/csv"
                    )
                else:
                    st.success("🎉 Şüpheli tesisat bulunamadı!")
                
                # Tüm sonuçlar
                st.subheader("📋 Tüm Sonuçlar")
                
                # Filtreleme seçenekleri
                filter_col1, filter_col2 = st.columns(2)
                
                with filter_col1:
                    suspicion_filter = st.selectbox(
                        "Şüpheli Durumu",
                        options=['Tümü', 'Şüpheli', 'Normal'],
                        index=0
                    )
                
                with filter_col2:
                    bina_filter = st.selectbox(
                        "Bina Numarası",
                        options=['Tümü'] + sorted(results_df['bina_no'].unique().tolist()),
                        index=0
                    )
                
                # Filtreleme uygula
                filtered_df = results_df.copy()
                
                if suspicion_filter != 'Tümü':
                    filtered_df = filtered_df[filtered_df['suspicion_level'] == suspicion_filter]
                
                if bina_filter != 'Tümü':
                    filtered_df = filtered_df[filtered_df['bina_no'] == bina_filter]
                
                # Sonuçları göster
                if not filtered_df.empty:
                    display_cols = ['tesisat_no', 'bina_no', 'kis_tuketim', 'yaz_tuketim', 
                                   'ortalama_tuketim', 'kis_trend', 'suspicion_level', 'anomaliler']
                    
                    filtered_display = filtered_df[display_cols].copy()
                    filtered_display.columns = ['Tesisat No', 'Bina No', 'Kış Tüketim', 
                                              'Yaz Tüketim', 'Ortalama Tüketim', 'Kış Trend',
                                              'Durum', 'Anomaliler']
                    
                    # Numeric columns için formatting
                    for col in ['Kış Tüketim', 'Yaz Tüketim', 'Ortalama Tüketim']:
                        filtered_display[col] = filtered_display[col].round(1)
                    
                    st.dataframe(
                        filtered_display,
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.warning("Filtreye uygun veri bulunamadı.")

else:
    st.info("👈 Lütfen sol panelden bir dosya yükleyin")
    
    # Örnek dosya formatı
    st.subheader("📄 Beklenen Dosya Formatı")
    st.write("Dosyanızda aşağıdaki sütunlar bulunmalıdır:")
    
    example_data = {
        'tesisat_no': ['T001', 'T002', 'T003'],
        'bina_no': ['B001', 'B001', 'B002'],
        '2024/1': [120, 25, 150],
        '2024/2': [110, 20, 140],
        '2024/3': [80, 15, 100],
        '2024/4': [50, 10, 60],
        '2024/5': [30, 8, 40],
        '2024/6': [25, 5, 35]
    }
    
    example_df = pd.DataFrame(example_data)
    st.dataframe(example_df, use_container_width=True)
    
    st.markdown("""
    **Dosya Formatı Açıklaması:**
    - **Tesisat Numarası**: Her tesisatın benzersiz kimlik numarası
    - **Bina Numarası**: Tesisatın bulunduğu binanın numarası
    - **Tarih Sütunları**: YYYY/M formatında (örn: 2024/1, 2024/2, ...)
    - **Tüketim Değerleri**: Aylık doğalgaz tüketimi (m³)
    """)

# Bilgi paneli
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Tespit Kriterleri")
st.sidebar.markdown(f"""
- **Kış Düşük Tüketim**: < {kis_tuketim_esigi} m³/ay
- **Bina Ortalaması**: %{bina_ort_dusuk_oran} düşük
- **Ani Düşüş**: %{ani_dusus_orani} düşüş
- **Kış-Yaz Farkı**: Çok az fark
- **Toplam Tüketim**: Çok düşük
- **Sıfır Tüketim**: 6+ ay sıfır
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Kullanım Bilgileri")
st.sidebar.markdown("""
1. CSV veya Excel dosyasını yükleyin
2. Tesisat ve bina sütunlarını seçin
3. Parametreleri ayarlayın
4. Analizi başlatın
5. Sonuçları inceleyin ve indirin
""")