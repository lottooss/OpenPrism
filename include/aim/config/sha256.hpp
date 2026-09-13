// include/aim/config/sha256.hpp
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>

namespace aim::config {

class Sha256 {
public:
    static constexpr std::size_t kBlockSize = 64;
    static constexpr std::size_t kDigestSize = 32;

    Sha256() noexcept {
        reset();
    }

    void reset() noexcept {
        state_[0] = 0x6a09e667UL;
        state_[1] = 0xbb67ae85UL;
        state_[2] = 0x3c6ef372UL;
        state_[3] = 0xa54ff53aUL;
        state_[4] = 0x510e527fUL;
        state_[5] = 0x9b05688cUL;
        state_[6] = 0x1f83d9abUL;
        state_[7] = 0x5be0cd19UL;
        count_ = 0;
        buffer_len_ = 0;
    }

    void update(const std::uint8_t* data, std::size_t len) noexcept {
        if (!data || len == 0) return;
        count_ += len;

        if (buffer_len_ > 0) {
            const std::size_t needed = kBlockSize - buffer_len_;
            if (len >= needed) {
                for (std::size_t i = 0; i < needed; ++i) {
                    buffer_[buffer_len_ + i] = data[i];
                }
                transform(buffer_.data());
                data += needed;
                len -= needed;
                buffer_len_ = 0;
            } else {
                for (std::size_t i = 0; i < len; ++i) {
                    buffer_[buffer_len_ + i] = data[i];
                }
                buffer_len_ += len;
                return;
            }
        }

        while (len >= kBlockSize) {
            transform(data);
            data += kBlockSize;
            len -= kBlockSize;
        }

        for (std::size_t i = 0; i < len; ++i) {
            buffer_[i] = data[i];
        }
        buffer_len_ = len;
    }

    void update(std::string_view sv) noexcept {
        update(reinterpret_cast<const std::uint8_t*>(sv.data()), sv.size());
    }

    [[nodiscard]] std::array<std::uint8_t, kDigestSize> finalize() noexcept {
        std::array<std::uint8_t, kDigestSize> digest{};

        const std::uint64_t total_bits = count_ * 8ULL;
        buffer_[buffer_len_++] = 0x80;

        if (buffer_len_ > 56) {
            for (std::size_t i = buffer_len_; i < kBlockSize; ++i) {
                buffer_[i] = 0;
            }
            transform(buffer_.data());
            buffer_len_ = 0;
        }

        for (std::size_t i = buffer_len_; i < 56; ++i) {
            buffer_[i] = 0;
        }

        for (int i = 7; i >= 0; --i) {
            buffer_[56 + static_cast<std::size_t>(7 - i)] = static_cast<std::uint8_t>((total_bits >> (i * 8)) & 0xFF);
        }

        transform(buffer_.data());

        for (std::size_t i = 0; i < 8; ++i) {
            digest[i * 4 + 0] = static_cast<std::uint8_t>((state_[i] >> 24) & 0xFF);
            digest[i * 4 + 1] = static_cast<std::uint8_t>((state_[i] >> 16) & 0xFF);
            digest[i * 4 + 2] = static_cast<std::uint8_t>((state_[i] >> 8) & 0xFF);
            digest[i * 4 + 3] = static_cast<std::uint8_t>(state_[i] & 0xFF);
        }

        reset();
        return digest;
    }

    [[nodiscard]] std::string finalize_hex() {
        const auto digest = finalize();
        static constexpr char hex_chars[] = "0123456789abcdef";
        std::string hex(kDigestSize * 2, '0');
        for (std::size_t i = 0; i < kDigestSize; ++i) {
            hex[i * 2 + 0] = hex_chars[(digest[i] >> 4) & 0x0F];
            hex[i * 2 + 1] = hex_chars[digest[i] & 0x0F];
        }
        return hex;
    }

    [[nodiscard]] static std::string hash_string(std::string_view sv) {
        Sha256 hasher;
        hasher.update(sv);
        return hasher.finalize_hex();
    }

    [[nodiscard]] static std::string hash_file(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            return "";
        }
        Sha256 hasher;
        std::array<char, 65536> buffer{};
        while (file.read(buffer.data(), static_cast<std::streamsize>(buffer.size())) || file.gcount() > 0) {
            hasher.update(reinterpret_cast<const std::uint8_t*>(buffer.data()), static_cast<std::size_t>(file.gcount()));
        }
        return hasher.finalize_hex();
    }

private:
    static constexpr std::uint32_t rotr(std::uint32_t x, std::uint32_t n) noexcept {
        return (x >> n) | (x << (32 - n));
    }

    static constexpr std::uint32_t ch(std::uint32_t x, std::uint32_t y, std::uint32_t z) noexcept {
        return (x & y) ^ (~x & z);
    }

    static constexpr std::uint32_t maj(std::uint32_t x, std::uint32_t y, std::uint32_t z) noexcept {
        return (x & y) ^ (x & z) ^ (y & z);
    }

    static constexpr std::uint32_t ep0(std::uint32_t x) noexcept {
        return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
    }

    static constexpr std::uint32_t ep1(std::uint32_t x) noexcept {
        return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
    }

    static constexpr std::uint32_t sig0(std::uint32_t x) noexcept {
        return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
    }

    static constexpr std::uint32_t sig1(std::uint32_t x) noexcept {
        return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
    }

    void transform(const std::uint8_t* chunk) noexcept {
        static constexpr std::uint32_t k[64] = {
            0x428a2f98UL, 0x71374491UL, 0xb5c0fbcfUL, 0xe9b5dba5UL,
            0x3956c25bUL, 0x59f111f1UL, 0x923f82a4UL, 0xab1c5ed5UL,
            0xd807aa98UL, 0x12835b01UL, 0x243185beUL, 0x550c7dc3UL,
            0x72be5d74UL, 0x80deb1feUL, 0x9bdc06a7UL, 0xc19bf174UL,
            0xe49b69c1UL, 0xefbe4786UL, 0x0fc19dc6UL, 0x240ca1ccUL,
            0x2de92c6fUL, 0x4a7484aaUL, 0x5cb0a9dcUL, 0x76f988daUL,
            0x983e5152UL, 0xa831c66dUL, 0xb00327c8UL, 0xbf597fc7UL,
            0xc6e00bf3UL, 0xd5a79147UL, 0x06ca6351UL, 0x14292967UL,
            0x27b70a85UL, 0x2e1b2138UL, 0x4d2c6dfcUL, 0x53380d13UL,
            0x650a7354UL, 0x766a0abbUL, 0x81c2c92eUL, 0x92722c85UL,
            0xa2bfe8a1UL, 0xa81a664bUL, 0xc24b8b70UL, 0xc76c51a3UL,
            0xd192e819UL, 0xd6990624UL, 0xf40e3585UL, 0x106aa070UL,
            0x19a4c116UL, 0x1e376c08UL, 0x2748774cUL, 0x34b0bcb5UL,
            0x391c0cb3UL, 0x4ed8aa4aUL, 0x5b9cca4fUL, 0x682e6ff3UL,
            0x748f82eeUL, 0x78a5636fUL, 0x84c87814UL, 0x8cc70208UL,
            0x90befffaUL, 0xa4506cebUL, 0xbef9a3f7UL, 0xc67178f2UL
        };

        std::uint32_t w[64];
        for (std::size_t i = 0; i < 16; ++i) {
            w[i] = (static_cast<std::uint32_t>(chunk[i * 4 + 0]) << 24) |
                   (static_cast<std::uint32_t>(chunk[i * 4 + 1]) << 16) |
                   (static_cast<std::uint32_t>(chunk[i * 4 + 2]) << 8) |
                   (static_cast<std::uint32_t>(chunk[i * 4 + 3]));
        }
        for (std::size_t i = 16; i < 64; ++i) {
            w[i] = sig1(w[i - 2]) + w[i - 7] + sig0(w[i - 15]) + w[i - 16];
        }

        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];

        for (std::size_t i = 0; i < 64; ++i) {
            const std::uint32_t t1 = h + ep1(e) + ch(e, f, g) + k[i] + w[i];
            const std::uint32_t t2 = ep0(a) + maj(a, b, c);
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }

        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{};
    std::uint64_t count_{0};
    std::array<std::uint8_t, kBlockSize> buffer_{};
    std::size_t buffer_len_{0};
};

} // namespace aim::config
