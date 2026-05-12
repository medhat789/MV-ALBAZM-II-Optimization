#!/usr/bin/env python3
"""
Enhanced Data Processor for M/V Al-bazm II
Processes FAOP-EOSP pairs and ECDIS weather data
"""

import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDataProcessor:
    """Process comprehensive voyage and weather data"""
    
    def __init__(self):
        self.voyage_data = None
        self.weather_data = None
        self.engine_power_kw = 5920  # Given by user
    
    def load_rob_data(self, filepath='/app/backend/OfficialROB2024-NOV2025.csv'):
        """Load and process Official ROB data with FAOP-EOSP pairs"""
        logger.info("📊 Loading Official ROB 2024-NOV 2025 data...")
        
        # Read CSV with proper encoding
        for encoding in ['latin1', 'iso-8859-1', 'cp1252', 'utf-8']:
            try:
                df = pd.read_csv(filepath, encoding=encoding, header=None)
                logger.info(f"Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        # Find the header row (contains "Event", "Date", etc.)
        header_row_idx = None
        for idx, row in df.iterrows():
            if any('Event' in str(cell) or 'E v e n t' in str(cell) for cell in row):
                header_row_idx = idx
                break
        
        if header_row_idx is None:
            logger.error("Could not find header row")
            return pd.DataFrame()
        
        # Extract column names from header row
        headers = df.iloc[header_row_idx].tolist()
        
        # Clean header names
        clean_headers = []
        for h in headers:
            if pd.isna(h):
                clean_headers.append('unnamed')
            else:
                # Remove spaces between letters
                h = str(h).replace(' ', '')
                clean_headers.append(h)
        
        # Get data starting from row after header
        data_df = df.iloc[header_row_idx + 1:].copy()
        data_df.columns = clean_headers
        
        # Reset index
        data_df = data_df.reset_index(drop=True)
        
        # Filter only rows with Event data (FAOP or EOSP)
        data_df = data_df[data_df['Event'].notna() & (data_df['Event'] != '')]
        data_df = data_df[data_df['Event'].str.contains('FAOP|EOSP', na=False)]
        
        logger.info(f"Found {len(data_df)} event records")
        
        # Process FAOP-EOSP pairs
        voyages = self._process_faop_eosp_pairs(data_df)
        
        logger.info(f"✅ Processed {len(voyages)} complete voyages")
        
        self.voyage_data = voyages
        return voyages
    
    def _process_faop_eosp_pairs(self, df):
        """Process FAOP-EOSP pairs to create complete voyage records"""
        voyages = []
        
        i = 0
        while i < len(df) - 1:
            current_row = df.iloc[i]
            next_row = df.iloc[i + 1]
            
            # Check if this is a FAOP-EOSP pair
            if 'FAOP' in str(current_row['Event']) and 'EOSP' in str(next_row['Event']):
                # Extract voyage data from EOSP row
                try:
                    voyage = self._extract_voyage_data(current_row, next_row)
                    if voyage:
                        voyages.append(voyage)
                except Exception as e:
                    logger.warning(f"Error processing voyage at index {i}: {e}")
                
                i += 2  # Skip to next FAOP
            else:
                i += 1
        
        return pd.DataFrame(voyages)
    
    def _extract_voyage_data(self, faop_row, eosp_row):
        """Extract voyage data from FAOP-EOSP pair"""
        
        # Parse date
        date_str = str(eosp_row['Date'])
        try:
            voyage_date = pd.to_datetime(date_str, format='%d-%b-%Y', errors='coerce')
            if pd.isna(voyage_date):
                voyage_date = pd.to_datetime(date_str, errors='coerce')
        except:
            voyage_date = None
        
        # Extract trip duration (HRS column)
        hrs = self._clean_numeric(eosp_row['HRS'])
        if not hrs or hrs <= 0:
            return None
        
        # Extract place/route
        place = str(eosp_row['Place']) if 'Place' in eosp_row else ''
        route = self._classify_route(place)
        
        # Extract slip
        slip = self._clean_numeric(eosp_row['Slip'])
        
        # Extract distance
        distance = self._clean_numeric(eosp_row.get('TotalDistance(NM)', 0))
        
        # Extract average speed
        avg_speed = self._clean_numeric(eosp_row.get('AvgSpeed(KNOTS)', 0))
        
        # Extract total fuel consumption
        total_foc = self._clean_numeric(eosp_row.get('TotalFOC(MT)', 0))
        if not total_foc or total_foc <= 0:
            return None
        
        # Parse FOC ME/GE (format: "7.3 / 1" or "7.3/1.0")
        foc_me_ge_str = str(eosp_row.get('FOCME/GE(MT)', ''))
        me_foc, ge_foc = self._parse_foc_me_ge(foc_me_ge_str)
        
        # Extract engine load percentage
        engine_load_str = str(eosp_row.get('Engine', ''))
        engine_load = self._parse_percentage(engine_load_str)
        
        # Extract RPM
        rpm = self._clean_numeric(eosp_row.get('RPM', None))
        
        return {
            'date': voyage_date,
            'trip_hours': hrs,
            'place': place,
            'route': route,
            'slip': slip if slip else 0,
            'distance_nm': distance if distance else 0,
            'speed_knots': avg_speed if avg_speed else 0,
            'fuel_mt': total_foc,
            'me_foc_mt': me_foc,
            'ge_foc_mt': ge_foc,
            'engine_load_pct': engine_load if engine_load else 50,
            'rpm': rpm
        }
    
    def _clean_numeric(self, value):
        """Clean and convert to numeric"""
        if pd.isna(value):
            return None
        
        try:
            # Remove any non-numeric characters except decimal point and minus
            cleaned = re.sub(r'[^\d.-]', '', str(value))
            if cleaned:
                return float(cleaned)
        except:
            pass
        
        return None
    
    def _parse_foc_me_ge(self, foc_str):
        """Parse FOC ME/GE format like '7.3 / 1' or '7.3/1.0'"""
        me_foc = 0
        ge_foc = 0
        
        if pd.isna(foc_str) or str(foc_str).strip() == '':
            return me_foc, ge_foc
        
        # Split by /
        parts = str(foc_str).split('/')
        
        if len(parts) >= 2:
            me_foc = self._clean_numeric(parts[0])
            ge_foc = self._clean_numeric(parts[1])
        elif len(parts) == 1:
            # Only ME value provided
            me_foc = self._clean_numeric(parts[0])
        
        return me_foc if me_foc else 0, ge_foc if ge_foc else 0
    
    def _parse_percentage(self, pct_str):
        """Parse percentage like '53%' or '53'"""
        if pd.isna(pct_str):
            return None
        
        # Remove % sign and clean
        cleaned = str(pct_str).replace('%', '').strip()
        return self._clean_numeric(cleaned)
    
    def _classify_route(self, place_text):
        """Classify route from place description"""
        if pd.isna(place_text):
            return 'Unknown'
        
        place = str(place_text).upper()
        
        # Check for direction indicators
        if 'KP' in place and 'RWS' in place:
            if '→' in place:
                if 'KP' in place.split('→')[0]:
                    return 'Khalifa_to_Ruwais'
                else:
                    return 'Ruwais_to_Khalifa'
            elif 'TO' in place:
                if 'KP' in place.split('TO')[0]:
                    return 'Khalifa_to_Ruwais'
                else:
                    return 'Ruwais_to_Khalifa'
        
        # Check arrival location
        if 'ARRIVAL' in place or 'ARRIVAL' in place:
            if 'RWS' in place or 'RUWAIS' in place:
                return 'Khalifa_to_Ruwais'
            elif 'KP' in place or 'KHL' in place or 'KHALIFA' in place:
                return 'Ruwais_to_Khalifa'
        
        return 'Unknown'
    
    def fill_missing_rpm(self):
        """Fill missing RPM values with average"""
        if self.voyage_data is None or len(self.voyage_data) == 0:
            return
        
        # Calculate average RPM from non-null values
        valid_rpm = self.voyage_data['rpm'].dropna()
        
        if len(valid_rpm) > 0:
            avg_rpm = valid_rpm.mean()
            
            # Fill missing values
            missing_count = self.voyage_data['rpm'].isna().sum()
            self.voyage_data['rpm'] = self.voyage_data['rpm'].fillna(avg_rpm)
            
            logger.info(f"Filled {missing_count} missing RPM values with average: {avg_rpm:.1f}")
    
    def load_ecdis_weather_data(self, filepath='/app/backend/ecdis_weather_data.xlsm'):
        """Load ECDIS weather data (Jan 18 2025 - April 15 2025)"""
        logger.info("🌊 Loading ECDIS weather data...")
        
        try:
            # Try reading Excel file
            weather_df = pd.read_excel(filepath, engine='openpyxl')
            logger.info(f"Loaded {len(weather_df)} weather records")
            
            self.weather_data = weather_df
            return weather_df
            
        except Exception as e:
            logger.warning(f"Could not load ECDIS data: {e}")
            return pd.DataFrame()
    
    def get_processed_data_summary(self):
        """Get summary of processed data"""
        if self.voyage_data is None or len(self.voyage_data) == 0:
            return {}
        
        df = self.voyage_data
        
        return {
            'total_voyages': len(df),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d') if df['date'].notna().any() else 'N/A',
                'end': df['date'].max().strftime('%Y-%m-%d') if df['date'].notna().any() else 'N/A'
            },
            'routes': df['route'].value_counts().to_dict(),
            'fuel_consumption': {
                'min_mt': float(df['fuel_mt'].min()),
                'max_mt': float(df['fuel_mt'].max()),
                'mean_mt': float(df['fuel_mt'].mean()),
                'std_mt': float(df['fuel_mt'].std())
            },
            'operational': {
                'speed_range': f"{df['speed_knots'].min():.1f} - {df['speed_knots'].max():.1f} knots",
                'duration_range': f"{df['trip_hours'].min():.1f} - {df['trip_hours'].max():.1f} hours",
                'mean_speed': float(df['speed_knots'].mean()),
                'mean_duration': float(df['trip_hours'].mean())
            },
            'engine': {
                'rpm_range': f"{df['rpm'].min():.0f} - {df['rpm'].max():.0f}" if df['rpm'].notna().any() else 'N/A',
                'mean_load_pct': float(df['engine_load_pct'].mean()),
                'mean_rpm': float(df['rpm'].mean()) if df['rpm'].notna().any() else 0
            }
        }


if __name__ == "__main__":
    # Test the processor
    processor = EnhancedDataProcessor()
    
    # Load ROB data
    voyages = processor.load_rob_data()
    
    # Fill missing RPM
    processor.fill_missing_rpm()
    
    # Load weather data
    weather = processor.load_ecdis_weather_data()
    
    # Print summary
    summary = processor.get_processed_data_summary()
    
    print("\n" + "="*60)
    print("Enhanced Data Processing Summary")
    print("="*60)
    print(f"\n📊 Total Voyages: {summary['total_voyages']}")
    print(f"📅 Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"\n🗺️  Routes:")
    for route, count in summary['routes'].items():
        print(f"   {route}: {count} voyages")
    print(f"\n⛽ Fuel Consumption:")
    print(f"   Range: {summary['fuel_consumption']['min_mt']:.2f} - {summary['fuel_consumption']['max_mt']:.2f} MT")
    print(f"   Mean: {summary['fuel_consumption']['mean_mt']:.2f} MT")
    print(f"\n🚢 Operational:")
    print(f"   Speed Range: {summary['operational']['speed_range']}")
    print(f"   Duration Range: {summary['operational']['duration_range']}")
    print(f"\n🔧 Engine:")
    print(f"   RPM Range: {summary['engine']['rpm_range']}")
    print(f"   Mean Load: {summary['engine']['mean_load_pct']:.1f}%")
    
    # Save processed data
    if len(voyages) > 0:
        voyages.to_csv('/app/backend/processed_voyages.csv', index=False)
        print(f"\n✅ Saved {len(voyages)} processed voyages to processed_voyages.csv")
