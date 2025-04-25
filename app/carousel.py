import streamlit as st

def render_carousel():
    st.subheader("🎥 Recently Released Movies")

    # Placeholder images for the carousel
    carousel_images = [
        "https://via.placeholder.com/300x400?text=Movie+1",
        "https://via.placeholder.com/300x400?text=Movie+2",
        "https://via.placeholder.com/300x400?text=Movie+3",
    ]

    # Use slider to navigate between images
    image_index = st.slider("Slide to view movies", 0, len(carousel_images) - 1, 0)
    st.image(carousel_images[image_index], use_column_width=True)

if __name__ == "__main__":
    render_carousel()