#pragma once


class ImageMatrix {
public:
    ImageMatrix(unsigned char* data, size_t rows, size_t cols, size_t channels = 4) : 
        m_data(data, data + rows * cols * channels), m_rows(rows), m_cols(cols), m_channels(channels)
    {}

    unsigned char* data() { return &m_data[0]; }
    size_t rows() const { return m_rows; }
    size_t cols() const { return m_cols; }
    size_t channels() const { return m_channels; }

private:
    std::vector<unsigned char> m_data;
    size_t m_rows, m_cols, m_channels;
};
