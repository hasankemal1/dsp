#include <iostream>
#include <sfml/Graphics.hpp>
#include <imgui-SFML.h>
#include <imgui.h>
#include <implot.h>
#include <vector>
#include <cmath>
#include <complex>

const float PI = 3.14159265358979323846f;
const float sampling_rate = 4000.0f; // Hz
const float signalgen = 50000.0f; // Hz
const float duration = 0.2f;
const float nquistsample =2645.0f; // Hz    
std::vector<double> x1;
std::vector<double> x2;
std::vector<double> x3;


void handleevent(sf::RenderWindow& window) {
	
	while (const auto event=window.pollEvent()) {
		ImGui::SFML::ProcessEvent(window,*event);
		if (event->is<sf::Event::Closed>()) {
			window.close();
		}
	}
}



void imgurender(
    const std::vector<double>& origin,
    const std::vector<double>& sampled,
    const std::vector<double>& fft_origin,
    const std::vector<double>& fft_sampled,
    const std::vector<double>& filtre,
    const std::vector<double>& filtre1,
	const std::vector<double>& orginsampled,
	const std::vector<double>& quanti,
	const std::vector<double>& sampledorgin
)
{
    ImGui::SetNextWindowPos(ImVec2(0, 0)); // Sol üst köþe
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize); // Tüm pencere boyutu

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoCollapse;



    ImGui::Begin("sinyal grafikleri");
    ImVec2 contentSize = ImGui::GetContentRegionAvail();
    if (ImPlot::BeginPlot("nonsampled signal",contentSize)) {
        ImPlot::SetupAxesLimits(0.140, 0.160, -4.2, 5.8);
        ImPlot::PlotLine("nonsampled Signal", x1.data(), origin.data(), (int)origin.size());
        ImPlot::EndPlot();
    }
    if (ImPlot::BeginPlot("sampled signal",contentSize)) {
       ImPlot::SetupAxesLimits(0.140,0.160 , -5, 5);
        ImPlot::PlotLine("sampled Signal", x2.data(), sampled.data(), (int)sampled.size());
        ImPlot::EndPlot();
    }
    if (ImPlot::BeginPlot("FFT of sampled signal",contentSize)) {
        ImPlot::SetupAxesLimits(-2000, 2000, 0, 1);
        size_t N = fft_sampled.size();
        double df = sampling_rate / N ;
        std::vector<double> freqs(N);
        for (size_t i = 0; i < N; ++i) {
            freqs[i] = (static_cast<double>(i) - N / 2) * df;
        }
        std::vector<double> shifted(N);
        int half = N / 2;
        for (size_t i = 0; i < half; ++i) {
            shifted[i] = fft_sampled[i + half];
            shifted[i + half] = fft_sampled[i];
        }
        ImPlot::PlotLine("FFT of sampled Signal", freqs.data(), shifted.data(), (int)N);
        ImPlot::EndPlot();
    }
    if (ImPlot::BeginPlot("FFT of nonsampled signal",contentSize)) {
        ImPlot::SetupAxesLimits(-2000, 2000, 0, 1);
        size_t N = fft_origin.size();
        double df = signalgen / N;
        std::vector<double> freqs(N);
        for (size_t i = 0; i < N; ++i) {
            freqs[i] = (static_cast<double>(i) - N / 2) * df;
        }
        std::vector<double> shifted(N);
        int half = N / 2;
        for (size_t i = 0; i < half; ++i) {
            shifted[i] = fft_origin[i + half];
            shifted[i + half] = fft_origin[i];
        }

        ImPlot::PlotLine("FFT of nonsampled Signal", freqs.data(), shifted.data(), (int)N);
        ImPlot::EndPlot();
    }
    if (ImPlot::BeginPlot("2khz FIR Filtered Signal",contentSize)) {
        ImPlot::SetupAxesLimits(0.1,0.113 , -6, 6);
        ImPlot::PlotLine("FIR Filtered Signal", x2.data(), filtre.data(), (int)filtre.size());
        ImPlot::EndPlot();
    }
    if (ImPlot::BeginPlot("1.5khz FIR Filtered Signal",contentSize)) {
        ImPlot::SetupAxesLimits(0.1, 0.113, -6, 6);
        ImPlot::PlotLine("FIR Filtered Signal", x2.data(), filtre1.data(), (int)filtre1.size());
        ImPlot::EndPlot();
    }

    if (ImPlot::BeginPlot("nquist sampled signal fft",contentSize)) {
        ImPlot::SetupAxesLimits(-2000, 2000,-0.5 , 1.5);
		
        size_t N = orginsampled.size();
        double df = nquistsample / N;
        std::vector<double> freqs(N);
        for (size_t i = 0; i < N; ++i) {
            freqs[i] = (static_cast<double>(i) - N / 2) * df;
        }
        std::vector<double> shifted(N);
        int half = N / 2;
        for (size_t i = 0; i < half; ++i) {
            shifted[i] = orginsampled[i + half];
            shifted[i + half] = orginsampled[i];
        }
        ImPlot::PlotLine("nquist Signal sampled fft", freqs.data(), shifted.data(), (int)orginsampled.size());
        ImPlot::EndPlot();
    }
    if (ImPlot::BeginPlot("quantize", contentSize)) {


              ImPlot::SetupAxesLimits(0, quanti.size(), -30768, 35768);
       
              ImPlot::PlotBars("Quantize", quanti.data(), (int)quanti.size(), 1.0f);
        ImPlot::EndPlot();
    }
    if (ImPlot::BeginPlot("rebuild signal", contentSize)) {


        ImPlot::SetupAxesLimits(0.1,0.113 , -6, 6);

		ImPlot::PlotLine("rebuild signal", x2.data(), sampledorgin.data(), (int)sampledorgin.size());
        ImPlot::EndPlot();
    }

    ImGui::End();
}

void generatesignal(std::vector<double>& orgin, std::vector<double>& sampled,std::vector<double>&sampledorgin) {
    int N_org = int(signalgen * duration);
    int N_samp = int(sampling_rate * duration);
	int N_orginsamp = int(nquistsample * duration);
	
	sampledorgin.resize(N_orginsamp);
	x3.resize(N_orginsamp);
    for (int i = 0; i < N_orginsamp; i++) {
        double t = static_cast<double>(i) / nquistsample;
        sampledorgin[i] = (2 * std::cos(2 * PI * 400.0f * t)) + std::cos(2 * PI * 800 * t) + (-3 * std::sin(2 * PI * 1200 * t));
        x3[i] = t;}

    orgin.resize(N_org);
    x1.resize(N_org);
    for (int i = 0; i < N_org; ++i) {
        double t = static_cast<double>(i) / signalgen;
        orgin[i] = (2 * std::cos(2 * PI * 400.0f * t)) + std::cos(2 * PI * 800 * t) + (-3 * std::sin(2 * PI * 1200 * t));
        x1[i] = t;
    }
    sampled.resize(N_samp);
    x2.resize(N_samp);
    for (int i = 0; i < N_samp; i++) {
        double t = static_cast<double>(i) / sampling_rate;
        sampled[i] = (2 * std::cos(2 * PI * 400.0f * t)) + std::cos(2 * PI * 800 * t) + (-3 * std::sin(2 * PI * 1200 * t));
        x2[i] = t;
    }
}

void fft(std::vector<std::complex<double>>& a, bool inverse = false) {
    size_t n = a.size();
    if (n <= 1) return;
    std::vector<std::complex<double>> even(n / 2), odd(n / 2);
    for (size_t i = 0; i < n / 2; ++i) {
        even[i] = a[i * 2];
        odd[i] = a[i * 2 + 1];
    }
    fft(even, inverse);
    fft(odd, inverse);
    double angle = 2 * PI / n * (inverse ? 1 : -1);
    std::complex<double> w(1);
    std::complex<double> wn(std::cos(angle), std::sin(angle));
    for (size_t k = 0; k < n / 2; ++k) {
        std::complex<double> t = w * odd[k];
        a[k] = even[k] + t;
        a[k + n / 2] = even[k] - t;
        w *= wn;
    }
    if (inverse) {
        for (auto& x : a) x /= n;
    }
}

double sinc(double x) {
    if (x == 0.0) return 1.0;
    return std::sin(PI * x) / (PI * x);
}

std::vector<double> firfiltre(const std::vector<double>& x, double stopfreaq) {
    double normalizefreaq = stopfreaq / (sampling_rate/2);
    int N = (int)x.size();
    std::vector<double> h(N);
    int mid = N / 2;
    for (int i = 0; i < N; i++) {
        if (i == mid) {
            h[i] = normalizefreaq;
        }
        else {
            h[i] = normalizefreaq * sinc((i - mid)*normalizefreaq) ;
        }
    }
    std::vector<double> y(N, 0.0);
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < N; k++) {
            int idx = n - k;
            if (idx >= 0 && idx < N)
                y[n] += h[k] * x[idx];
        }
    }
    return y;
}

std::vector<double> quant(std::vector<double>& nonquad)  
{  
    int b = 16;
    int levels = pow(2,b)-1; // 65536 levels
   
    float max_value = *std::max_element(nonquad.begin(), nonquad.end());
    float min_value = *std::min_element(nonquad.begin(), nonquad.end());
    float range = max_value - min_value;
    float step = range / levels; // quantization step

    std::vector<double> y_quantized(nonquad.size());
    for (size_t i = 0; i < nonquad.size(); i++) {
        y_quantized[i] = std::round((nonquad[i]) / step);
    }

    return y_quantized;
}
std::vector<double> apply_equalizer(const std::vector<double>& signal) {
    int N = 51; // filtre uzunluðu
    std::vector<double> h(N);
    double Ts = 1.0 / sampling_rate * 2;
    int mid = N / 2;
    for (int i = 0; i < N; ++i) {
        double t = (i - mid) * Ts;
        h[i] = sinc(t / Ts); // Sadece klasik sinc çekirdeði
    }

    // Konvolüsyon
    std::vector<double> y(signal.size(), 0.0);
    for (size_t n = 0; n < signal.size(); ++n) {
        for (int k = 0; k < N; ++k) {
            if (n >= k)
                y[n] += h[k] * signal[n - k];
        }
    }
    return y;
}



int main() {
    sf::RenderWindow window(sf::VideoMode({ 800, 600 }), "sayýsal haberleþme ödev");
    if (!ImGui::SFML::Init(window)) {
        std::cerr << "Failed to initialize ImGui-SFML" << std::endl;
        return -1;
    }
    
    ImPlot::CreateContext();

    std::vector<double> orgin, sampled,orginsampled;
    generatesignal(orgin, sampled,orginsampled);

    // FFT için karmaþýk vektör oluþtur
    std::vector<std::complex<double>> a(sampled.begin(), sampled.end());
    fft(a);
    std::vector<double> real_parts1(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        real_parts1[i] = std::abs(a[i]) / (sampled.size());
    }

    std::vector<std::complex<double>> b(orgin.begin(), orgin.end());
    fft(b);
    std::vector<double> real_parts(b.size());
    for (size_t i = 0; i < b.size(); ++i) {
        real_parts[i] = std::abs(b[i]) / (orgin.size());
    }

	std::vector<std::complex<double>> c(orginsampled.begin(), orginsampled.end());
	fft(c);
	std::vector<double> real_parts2(c.size());
	for (size_t i = 0; i < c.size(); ++i) {
		real_parts2[i] = std::abs(c[i]) / ( orginsampled.size());
	}


	std::vector<double> quanti = quant(sampled);
   
	std::vector<double> sampledorgin = apply_equalizer(sampled);
    std::vector<double> filter = firfiltre(sampled, 2000.0f);
    std::vector<double> filter1 = firfiltre(sampled, 1500.0f);

    sf::Clock deltaClock;
    while (window.isOpen()) {
        handleevent(window);
        ImGui::SFML::Update(window, deltaClock.restart());
        imgurender(orgin, sampled, real_parts, real_parts1, filter, filter1,real_parts2,quanti,sampledorgin);
        window.clear();
        ImGui::SFML::Render(window);
        window.display();
    }
    ImGui::SFML::Shutdown();
    return 0;
}