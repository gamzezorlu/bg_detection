import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Streamlit sayfa konfigürasyonu
st.set_page_config(
    page_title="Doğalgaz Kaçak Kullanım Tespit Sistemi",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Doğalgaz Kaçak Kullanım Tespit Sistemi")
st.markdown("---")

# Excel dosyasını DataFrame'e dönüştürme fonksiyonu
def excel_to_dataframe(file_buffer):
    """Excel dosyasını okuyup DataFrame'e dönüştür"""
    try:
        # Excel dosyasını oku
        df = pd.read_excel(file_buffer, engine='openpyxl')
        
        # Sütun isimlerini temizle
        df.columns = df.columns.str.strip()
        
        # Ay sütunlarını belirle (tarih formatındaki sütunlar)
        ay_sutunlari = []
        for col in df.columns:
            if any(x in str(col) for x in ['2023', '2024', '/', '-']):
                ay_sutunlari.append(col)
        
        # Eğer ay sütunları bulunamazsa, numerik sütunları al (TN ve BN hariç)
        if not ay_sutunlari:
            ay_sutunlari = [col for col in df.columns if col not in ['TN', 'BN'] and pd.api.types.is_numeric_dtype(df[col])]
        
        # Virgülleri nokta ile değiştir ve sayısal verilere dönüştür
        for col in ay_sutunlari:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df, ay_sutunlari
    except Exception as e:
        st.error(f"Excel dosyası okuma hatası: {str(e)}")
        return None, None

# Risk puanı hesaplama fonksiyonu
def calculate_risk_score(df, ay_sutunlari):
    """Her tesisat için risk puanı hesapla"""
    
    # Sonuç dataframe'i oluştur
    risk_df = df[['TN', 'BN']].copy()
    
    # Ay sütunlarını sırala
    ay_sutunlari_sorted = sorted(ay_sutunlari)
    
    # Kış ve yaz aylarını belirle (genel olarak)
    kis_aylari = []
    yaz_aylari = []
    
    for ay in ay_sutunlari_sorted:
        ay_str = str(ay).lower()
        if any(x in ay_str for x in ['12', 'ocak', 'şubat', 'mart', '01', '02', '03']):
            kis_aylari.append(ay)
        elif any(x in ay_str for x in ['06', 'temmuz', 'ağustos', 'eylül', '07', '08', '09']):
            yaz_aylari.append(ay)
    
    # BN gruplarına göre istatistikler
    bn_stats = {}
    for bn in df['BN'].unique():
        bn_data = df[df['BN'] == bn]
        bn_means = []
        for col in ay_sutunlari:
            col_mean = bn_data[col].mean()
            if not pd.isna(col_mean):
                bn_means.append(col_mean)
        
        if bn_means:
            bn_stats[bn] = {
                'mean': np.mean(bn_means),
                'median': np.median(bn_means),
                'std': np.std(bn_means)
            }
        else:
            bn_stats[bn] = {'mean': 0, 'median': 0, 'std': 0}
    
    risk_scores = []
    risk_details = []
    ortalama_tuketimler = []
    volatilite_degerler = []
    sifir_oranlari = []
    
    for idx, row in df.iterrows():
        risk_score = 0
        details = []
        bn = row['BN']
        
        # Geçerli tüketim değerlerini al
        tuketim_degerleri = []
        for col in ay_sutunlari:
            if col in row.index and not pd.isna(row[col]) and row[col] >= 0:
                tuketim_degerleri.append(row[col])
        
        if not tuketim_degerleri:
            risk_scores.append(0)
            risk_details.append("Veri yetersiz")
            ortalama_tuketimler.append(0)
            volatilite_degerler.append(0)
            sifir_oranlari.append(0)
            continue
        
        # 1. Genel düşük kullanım puanı (BN grubuna göre)
        ortalama_tuketim = np.mean(tuketim_degerleri)
        ortalama_tuketimler.append(ortalama_tuketim)
        bn_ortalama = bn_stats[bn]['mean']
        
        if bn_ortalama > 0:
            düsük_kullanım_oranı = (bn_ortalama - ortalama_tuketim) / bn_ortalama * 100
            
            if 20 <= düsük_kullanım_oranı < 50:
                risk_score += 1
                details.append(f"Düşük kullanım (%{düsük_kullanım_oranı:.1f}): +1 puan")
            elif düsük_kullanım_oranı >= 50:
                risk_score += 2
                details.append(f"Çok düşük kullanım (%{düsük_kullanım_oranı:.1f}): +2 puan")
        
        # 2. Kış aylarında düşüş analizi
        if kis_aylari:
            kis_verileri = []
            for ay in kis_aylari:
                if ay in row.index and not pd.isna(row[ay]) and row[ay] >= 0:
                    kis_verileri.append(row[ay])
            
            if len(kis_verileri) >= 2:
                kis_max = max(kis_verileri)
                kis_min = min(kis_verileri)
                
                if kis_max > 0:
                    kis_dusuş_oranı = (kis_max - kis_min) / kis_max * 100
                    
                    if 10 <= kis_dusuş_oranı < 40:
                        risk_score += 1
                        details.append(f"Kış ayı düşüşü (%{kis_dusuş_oranı:.1f}): +1 puan")
                    elif kis_dusuş_oranı >= 40:
                        risk_score += 2
                        details.append(f"Kış ayı büyük düşüş (%{kis_dusuş_oranı:.1f}): +2 puan")
        
        # 3. Sıfır veya çok düşük değer sıklığı
        sifir_sayisi = 0
        toplam_veri = 0
        
        for col in ay_sutunlari:
            if col in row.index:
                toplam_veri += 1
                if pd.isna(row[col]) or row[col] == 0:
                    sifir_sayisi += 1
        
        if toplam_veri > 0:
            sifir_oranı = sifir_sayisi / toplam_veri * 100
            sifir_oranlari.append(sifir_oranı)
            
            if sifir_oranı >= 10:
                risk_score += 1
                details.append(f"Sıfır değer sıklığı (%{sifir_oranı:.1f}): +1 puan")
            if sifir_oranı >= 25:
                risk_score += 1
                details.append(f"Yüksek sıfır değer sıklığı (%{sifir_oranı:.1f}): +1 puan")
        else:
            sifir_oranlari.append(0)
        
        # 4. Ani değişimler (volatilite)
        if len(tuketim_degerleri) >= 3:
            tuketim_pozitif = [val for val in tuketim_degerleri if val > 0]
            if len(tuketim_pozitif) >= 3:
                değişim_katsayısı = np.std(tuketim_pozitif) / np.mean(tuketim_pozitif)
                volatilite_degerler.append(değişim_katsayısı)
                
                if değişim_katsayısı > 1.5:
                    risk_score += 1
                    details.append(f"Yüksek volatilite (CV: {değişim_katsayısı:.2f}): +1 puan")
            else:
                volatilite_degerler.append(0)
        else:
            volatilite_degerler.append(0)
        
        # 5. Mevsimsel pattern eksikliği
        if kis_aylari and yaz_aylari:
            kis_tuketim = []
            yaz_tuketim = []
            
            for ay in kis_aylari:
                if ay in row.index and not pd.isna(row[ay]) and row[ay] > 0:
                    kis_tuketim.append(row[ay])
            
            for ay in yaz_aylari:
                if ay in row.index and not pd.isna(row[ay]) and row[ay] > 0:
                    yaz_tuketim.append(row[ay])
            
            if len(kis_tuketim) >= 1 and len(yaz_tuketim) >= 1:
                kis_ort = np.mean(kis_tuketim)
                yaz_ort = np.mean(yaz_tuketim)
                
                if yaz_ort > 0:
                    mevsimsel_oran = kis_ort / yaz_ort
                    if mevsimsel_oran < 1.2:
                        risk_score += 1
                        details.append(f"Zayıf mevsimsel pattern ({mevsimsel_oran:.2f}): +1 puan")
        
        risk_scores.append(risk_score)
        risk_details.append("; ".join(details) if details else "Risk faktörü yok")
    
    # Sonuçları ekle
    risk_df['Risk_Puani'] = risk_scores
    risk_df['Risk_Detaylari'] = risk_details
    risk_df['Ortalama_Tuketim'] = ortalama_tuketimler
    risk_df['Volatilite'] = volatilite_degerler
    risk_df['Sifir_Orani'] = sifir_oranlari
    
    # Risk seviyesi belirleme
    risk_df['Risk_Seviyesi'] = pd.cut(
        risk_df['Risk_Puani'],
        bins=[-1, 0, 2, 4, 10],
        labels=['Düşük', 'Orta', 'Yüksek', 'Çok Yüksek']
    )
    
    return risk_df

# Excel dosyası oluşturma fonksiyonu
def create_excel_report(df, risk_df, ay_sutunlari):
    """Analiz sonuçlarını Excel formatında hazırla"""
    
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        
        # 1. Risk Analizi Sayfası
        risk_summary = risk_df.copy()
        risk_summary.to_excel(writer, sheet_name='Risk_Analizi', index=False)
        
        # 2. Yüksek Riskli Tesisatlar
        yuksek_risk = risk_df[risk_df['Risk_Seviyesi'].isin(['Yüksek', 'Çok Yüksek'])].copy()
        yuksek_risk = yuksek_risk.sort_values('Risk_Puani', ascending=False)
        yuksek_risk.to_excel(writer, sheet_name='Yuksek_Riskli', index=False)
        
        # 3. Detaylı Tüketim Verileri
        detayli_veri = df.merge(risk_df[['TN', 'Risk_Puani', 'Risk_Seviyesi']], on='TN', how='left')
        detayli_veri.to_excel(writer, sheet_name='Detayli_Veri', index=False)
        
        # 4. Özet İstatistikler
        ozet_stats = {
            'Metrik': [
                'Toplam Tesisat Sayısı',
                'Düşük Risk',
                'Orta Risk', 
                'Yüksek Risk',
                'Çok Yüksek Risk',
                'Ortalama Risk Puanı',
                'Maksimum Risk Puanı'
            ],
            'Değer': [
                len(risk_df),
                len(risk_df[risk_df['Risk_Seviyesi'] == 'Düşük']),
                len(risk_df[risk_df['Risk_Seviyesi'] == 'Orta']),
                len(risk_df[risk_df['Risk_Seviyesi'] == 'Yüksek']),
                len(risk_df[risk_df['Risk_Seviyesi'] == 'Çok Yüksek']),
                round(risk_df['Risk_Puani'].mean(), 2),
                risk_df['Risk_Puani'].max()
            ]
        }
        
        ozet_df = pd.DataFrame(ozet_stats)
        ozet_df.to_excel(writer, sheet_name='Ozet_Istatistikler', index=False)
        
        # 5. BN Grupları Analizi
        bn_analizi = risk_df.groupby('BN').agg({
            'Risk_Puani': ['count', 'mean', 'max'],
            'Ortalama_Tuketim': 'mean',
            'Volatilite': 'mean'
        }).round(2)
        
        bn_analizi.columns = ['Tesisat_Sayisi', 'Ortalama_Risk', 'Max_Risk', 'Ortalama_Tuketim', 'Ortalama_Volatilite']
        bn_analizi.reset_index().to_excel(writer, sheet_name='BN_Analizi', index=False)
    
    output.seek(0)
    return output

# Ana uygulama
def main():
    
    # Dosya yükleme
    st.subheader("📁 Excel Dosyası Yükleme")
    uploaded_file = st.file_uploader(
        "Doğalgaz tüketim verilerini içeren Excel dosyasını yükleyin",
        type=['xlsx', 'xls'],
        help="Dosya TN (Tesisat Numarası), BN (Bağlantı Nesnesi) sütunları ve aylık tüketim verilerini içermelidir."
    )
    
    if uploaded_file is not None:
        # Dosyayı oku
        with st.spinner("Excel dosyası okunuyor..."):
            df, ay_sutunlari = excel_to_dataframe(uploaded_file)
        
        if df is None:
            st.error("Dosya okunamadı. Lütfen dosya formatını kontrol edin.")
            return
        
        # Veri önizleme
        st.subheader("📋 Veri Önizleme")
        st.write(f"**Toplam Tesisat Sayısı:** {len(df)}")
        st.write(f"**Tespit Edilen Ay Sütunları:** {len(ay_sutunlari)}")
        st.write(f"**Ay Sütunları:** {', '.join(map(str, ay_sutunlari))}")
        
        # İlk 5 satırı göster
        st.write("**İlk 5 Satır:**")
        st.dataframe(df.head())
        
        # Analiz butonu
        if st.button("🔍 Analizi Başlat", type="primary"):
            
            # Risk puanı hesapla
            with st.spinner("Risk puanları hesaplanıyor..."):
                risk_df = calculate_risk_score(df, ay_sutunlari)
            
            # Sidebar - Filtreler
            st.sidebar.header("🔧 Filtreler")
            
            # BN filtresi
            bn_secenekleri = ['Tümü'] + sorted(df['BN'].unique().tolist())
            secili_bn = st.sidebar.selectbox("Bağlantı Nesnesi (BN)", bn_secenekleri)
            
            # Filtreleme
            görüntülenen_risk_df = risk_df.copy()
            görüntülenen_df = df.copy()
            
            if secili_bn != 'Tümü':
                görüntülenen_risk_df = görüntülenen_risk_df[görüntülenen_risk_df['BN'] == secili_bn]
                görüntülenen_df = görüntülenen_df[görüntülenen_df['BN'] == secili_bn]
            
            # Risk seviyesi filtresi
            risk_seviyesi = st.sidebar.multiselect(
                "Risk Seviyesi",
                options=['Düşük', 'Orta', 'Yüksek', 'Çok Yüksek'],
                default=['Düşük', 'Orta', 'Yüksek', 'Çok Yüksek']
            )
            
            if risk_seviyesi:
                görüntülenen_risk_df = görüntülenen_risk_df[görüntülenen_risk_df['Risk_Seviyesi'].isin(risk_seviyesi)]
            
            # Ana dashboard
            st.subheader("📊 Analiz Sonuçları")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Toplam Tesisat", len(df))
            
            with col2:
                yuksek_risk = len(risk_df[risk_df['Risk_Seviyesi'].isin(['Yüksek', 'Çok Yüksek'])])
                st.metric("Yüksek Risk", yuksek_risk)
            
            with col3:
                ortalama_risk = risk_df['Risk_Puani'].mean()
                st.metric("Ortalama Risk Puanı", f"{ortalama_risk:.1f}")
            
            with col4:
                max_risk = risk_df['Risk_Puani'].max()
                st.metric("Maksimum Risk Puanı", max_risk)
            
            # Grafikler
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk seviyesi dağılımı
                risk_dagılım = risk_df['Risk_Seviyesi'].value_counts()
                fig_pie = px.pie(
                    values=risk_dagılım.values,
                    names=risk_dagılım.index,
                    title="Risk Seviyesi Dağılımı",
                    color_discrete_map={
                        'Düşük': '#2E8B57',
                        'Orta': '#FFD700', 
                        'Yüksek': '#FF6347',
                        'Çok Yüksek': '#DC143C'
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Risk puanı histogramı
                fig_hist = px.histogram(
                    risk_df,
                    x='Risk_Puani',
                    nbins=20,
                    title="Risk Puanı Dağılımı",
                    color_discrete_sequence=['#1f77b4']
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # Yüksek riskli tesisatlar tablosu
            st.subheader("⚠️ Risk Analizi Tablosu")
            
            # Sıralama seçenekleri
            siralama = st.selectbox(
                "Sıralama",
                options=['Risk Puanı (Yüksek → Düşük)', 'Risk Puanı (Düşük → Yüksek)', 'Tesisat Numarası']
            )
            
            if siralama == 'Risk Puanı (Yüksek → Düşük)':
                görüntülenen_risk_df = görüntülenen_risk_df.sort_values('Risk_Puani', ascending=False)
            elif siralama == 'Risk Puanı (Düşük → Yüksek)':
                görüntülenen_risk_df = görüntülenen_risk_df.sort_values('Risk_Puani', ascending=True)
            else:
                görüntülenen_risk_df = görüntülenen_risk_df.sort_values('TN')
            
            # Tablo gösterimi
            st.dataframe(
                görüntülenen_risk_df[['TN', 'BN', 'Risk_Puani', 'Risk_Seviyesi', 'Ortalama_Tuketim', 'Volatilite', 'Sifir_Orani', 'Risk_Detaylari']],
                use_container_width=True,
                column_config={
                    'TN': st.column_config.NumberColumn('Tesisat No'),
                    'BN': st.column_config.NumberColumn('Bağlantı Nesnesi'),
                    'Risk_Puani': st.column_config.NumberColumn('Risk Puanı'),
                    'Risk_Seviyesi': st.column_config.SelectboxColumn('Risk Seviyesi'),
                    'Ortalama_Tuketim': st.column_config.NumberColumn('Ortalama Tüketim', format="%.2f"),
                    'Volatilite': st.column_config.NumberColumn('Volatilite', format="%.2f"),
                    'Sifir_Orani': st.column_config.NumberColumn('Sıfır Oranı (%)', format="%.1f"),
                    'Risk_Detaylari': st.column_config.TextColumn('Risk Detayları')
                }
            )
            
            # Excel raporu oluştur ve indir
            st.subheader("📥 Rapor İndirme")
            
            excel_buffer = create_excel_report(df, risk_df, ay_sutunlari)
            
            # İndirme butonu
            st.download_button(
                label="📊 Excel Raporu İndir",
                data=excel_buffer,
                file_name=f"dogalgaz_risk_analizi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
            
            # Rapor içeriği bilgisi
            st.info("""
            **Excel Raporu İçeriği:**
            - 📋 **Risk Analizi**: Tüm tesisatların risk puanları ve detayları
            - ⚠️ **Yüksek Riskli**: Yüksek ve çok yüksek risk seviyesindeki tesisatlar
            - 📊 **Detaylı Veri**: Orijinal tüketim verileri + risk puanları
            - 📈 **Özet İstatistikler**: Genel analiz sonuçları
            - 🏢 **BN Analizi**: Bağlantı nesnesi grupları analizi
            """)
            
            # Detaylı analiz
            st.subheader("🔍 Detaylı Tesisat Analizi")
            
            # Tesisat seçimi
            secili_tesisat = st.selectbox(
                "Analiz edilecek tesisatı seçin",
                options=sorted(görüntülenen_risk_df['TN'].tolist())
            )
            
            if secili_tesisat:
                # Seçili tesisat verilerini al
                tesisat_verisi = df[df['TN'] == secili_tesisat].iloc[0]
                tesisat_risk = risk_df[risk_df['TN'] == secili_tesisat].iloc[0]
                
                # Tesisat bilgileri
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Tesisat No", secili_tesisat)
                
                with col2:
                    st.metric("Bağlantı Nesnesi", tesisat_verisi['BN'])
                
                with col3:
                    st.metric("Risk Puanı", tesisat_risk['Risk_Puani'])
                
                with col4:
                    st.metric("Risk Seviyesi", tesisat_risk['Risk_Seviyesi'])
                
                # Tüketim grafiği
                tuketim_verileri = []
                for ay in ay_sutunlari:
                    if ay in tesisat_verisi.index and not pd.isna(tesisat_verisi[ay]):
                        tuketim_verileri.append({
                            'Ay': str(ay),
                            'Tuketim': tesisat_verisi[ay]
                        })
                
                if tuketim_verileri:
                    tuketim_df = pd.DataFrame(tuketim_verileri)
                    
                    fig_line = px.line(
                        tuketim_df,
                        x='Ay',
                        y='Tuketim',
                        title=f"Tesisat {secili_tesisat} - Aylık Doğalgaz Tüketimi",
                        markers=True
                    )
                    fig_line.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_line, use_container_width=True)
                
                # Risk detayları
                st.subheader("Risk Analizi Detayları")
                st.write(tesisat_risk['Risk_Detaylari'])
    
    else:
        st.info("📁 Analiz için Excel dosyasını yükleyin")
        st.markdown("""
        **Dosya Formatı:**
        - Excel dosyası (.xlsx veya .xls)
        - **TN** sütunu: Tesisat numaraları
        - **BN** sütunu: Bağlantı nesnesi numaraları  
        - **Ay sütunları**: Aylık tüketim verileri (örn: 2023/09, 2023/10, vb.)
        
        **Örnek Dosya Yapısı:**
        ```
        TN      | BN        | 2023/09 | 2023/10 | 2023/11 | ...
        1000001 | 1000001619| 12.73   | 13.98   | 15.88   | ...
        1000002 | 1000001619| 32.75   | 25.17   | 46.70   | ...
        ```
        """)

if __name__ == "__main__":
    main()
