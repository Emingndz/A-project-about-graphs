Burası güncellemelerden etkilenmemek için kullanılacaktır.

Şimdi bi şeyler denemeden önceki son checkout gibi düşün değiştirdikçe commit edicem

Son adımda : on_click_stop fonksiyonundan, tek tek kenarları remove() etme adımları çıkarıldı.
Fonksiyon, grafiği tamamen ax.clear() ile temizleyip sonra ilk haline yeniden çizer hale getirildi.
drawn_edges.clear() çağrısı, ek path çizimleri vb. referansları da sıfırlamak için korundu.