#!/usr/bin/env python3
"""
Eye Mouse - Entry Point
Gaze-controlled cursor application
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    try:
        from src.gui.main_window import main
        main()
    except Exception as e:
        import traceback
        error_msg = f"HATA OLUSTU:\n{str(e)}\n\nDETAYLAR:\n{traceback.format_exc()}"
        
        # Log dosyasini uygulamanin yanina kaydet
        if getattr(sys, 'frozen', False):
            # Exe ise exe'nin yani
            application_path = os.path.dirname(sys.executable)
            # macOS .app icindeyse bozulabilir, ama kullanici bunu istiyor
        else:
            # Script ise script yani
            application_path = os.path.dirname(os.path.abspath(__file__))
            
        log_path = os.path.join(application_path, "Cyclops_Error_Log.txt")
        
        # Hata dosyasina yaz
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(error_msg)
            
        # Eger PyQt yukluyse popup goster (GUI cokmeden once)
        try:
            from PyQt6.QtWidgets import QApplication, QMessageBox
            if QApplication.instance():
                QMessageBox.critical(None, "Kritik Hata", f"Uygulama coktu!\nHata raporu masaüstüne kaydedildi:\n{log_path}\n\nHata: {str(e)}")
        except:
            pass
        
        # Konsola da yaz
        print(error_msg)
        sys.exit(1)
