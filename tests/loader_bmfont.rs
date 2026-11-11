//! BMFont conformance: one font as a text and an XML descriptor, both
//! clearing the floor and decoding the same, with its metrics, pages, glyph
//! channels and kerning intact.

#![cfg(feature = "bmfont")]

use std::path::PathBuf;

use viewport_lib_io::loaders::bmfont;
use viewport_lib_io::types::{BitmapFontData, GlyphChannels, PixelRect, TextureSource};
use viewport_lib_io_testkit::{conformance, synth};

fn decode(xml: bool) -> (PathBuf, BitmapFontData) {
    let path = synth::temp_path("bmfont", "font.fnt");
    synth::write(&path, synth::bmfont(xml));
    let font = conformance::bitmap_font(&format!("bmfont xml={xml}"), || {
        bmfont::bitmap_font_from_path(&path)
    });
    (path, font)
}

#[test]
fn text_and_xml_descriptors_decode_the_same() {
    let without_pages = |mut f: BitmapFontData| {
        f.pages.clear();
        f
    };
    assert_eq!(
        without_pages(decode(false).1),
        without_pages(decode(true).1)
    );
}

#[test]
fn metrics_and_pages_come_through_in_page_order() {
    let (path, font) = decode(false);
    let dir = path.parent().unwrap();
    assert_eq!(
        (font.face.as_str(), font.size, font.bold),
        ("Pixel Sans", -16, true)
    );
    assert_eq!(
        (font.line_height, font.base, font.page_size),
        (18, 14, [64, 64])
    );
    assert_eq!((font.padding, font.spacing), ([1, 2, 3, 4], [1, 1]));
    assert!(font.packed);
    assert_eq!(
        font.pages,
        [
            TextureSource::File(dir.join("glyphs_0.png")),
            TextureSource::File(dir.join("glyphs_1.png")),
        ],
        "sorted by page id, not file order"
    );
}

#[test]
fn glyphs_carry_their_rect_offsets_page_and_channel() {
    let (_, font) = decode(true);
    let a = font.glyphs[&65];
    assert_eq!((a.page, a.rect), (0, PixelRect::new(0, 0, 10, 12)));
    assert_eq!((a.offset, a.advance), ([1, -2], 11));
    assert_eq!(a.uv(font.page_size), [0.0, 0.0, 10.0 / 64.0, 12.0 / 64.0]);

    let only = |red, green, blue| GlyphChannels {
        red,
        green,
        blue,
        alpha: false,
    };
    assert_eq!(a.channels, only(true, false, false));
    assert_eq!(font.glyphs[&66].channels, only(false, true, false));
    assert_eq!(font.glyphs[&67].channels, only(false, false, true));
    assert_eq!(font.glyphs[&32].channels, GlyphChannels::ALL);
    assert_eq!(font.glyphs[&32].page, 1);

    assert_eq!(font.kernings.get(&(65, 66)), Some(&-1));
    assert_eq!(font.kernings.len(), 1);
}
