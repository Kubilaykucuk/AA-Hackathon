# Dağıtım Talimatları

## Ortam Kurulumu

Projeyi çalıştırmadan önce sisteminizde Python'un kurulu olduğundan emin olun. Bu proje Python 3.8 ile geliştirilmiştir, ancak Python 3.6 ve üzeri sürümlerle uyumlu olmalıdır.

### Adım 1: Depoyu Kopyala

Git kullanarak bu depoyu yerel makinenize kopyalayın:

git clone <depo-url>
cd <depo-dizini>

`<depo-url>`'i Git deposunun URL'si ile ve `<depo-dizini>`'ni depo kopyalandığında oluşturulan klasörün adı ile değiştirin.

### Adım 2: Sanal Ortam Oluştur (İsteğe Bağlı)

Bağımlılıkları yönetmek için bir sanal ortam oluşturmanız önerilir:

python -m venv venv

Sanal ortamı etkinleştirin:

- Windows'ta:
  .\venv\Scripts\activate

- macOS ve Linux'ta:
  source venv/bin/activate

### Adım 3: Bağımlılıkları Yükle

Pip kullanarak gerekli Python modüllerini yükleyin:

pip install transformers tkinter nltk networkx numpy

Not: `tkinter` genellikle Python ile birlikte gelir, bu yüzden ayrıca yüklemeniz gerekmeyebilir. `tkinter` ile ilgili herhangi bir sorunla karşılaşırsanız, Python kurulumunuzda dahil edildiğinden emin olun.

### Adım 4: NLTK Verilerini İndir

Bazı NLTK işlevselliği ek veri gerektirir. Gerekli veri setlerini indirmek için aşağıdaki Python komutunu çalıştırın:

```python
import nltk
nltk.download('punkt')

### Adım 5: Uygulamayı Çalıştır

Tüm bağımlılıkları yükledikten sonra, uygulamayı çalıştırabilirsiniz:

python main.py

### Sorun Giderme

Bağımlılıklarla ilgili herhangi bir sorunla karşılaşırsanız, pip'inizin güncel olduğundan emin olun:

pip install --upgrade pip

Belirli paketlerle ilgili sorunlar için, en güncel kurulum talimatları için resmi dokümantasyonlarına başvurun.

Depo URL'si ve dizini ihtiyacınıza göre ayarlayın. Bu markdown dosyası, Python ve komut satırı işlemleriyle temel bir aşinalık varsayar.
