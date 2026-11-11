// SPDX-License-Identifier: GPL-2.0-or-later
//! Bounded, dependency-free XML property lists for the firmware loader.

use alloc::{collections::BTreeMap, string::String, vec::Vec};
use core::{fmt, str};

pub type Dictionary = BTreeMap<String, Value>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Dictionary(Dictionary),
    Array(Vec<Value>),
    String(String),
    Integer(i128),
    Boolean(bool),
    Data(Vec<u8>),
}

/// Check this before allocating a buffer for an on-disk configuration.
pub const MAX_BYTES: usize = 128 * 1024;
const MAX_DEPTH: usize = 24;
const MAX_NODES: usize = 8192;
const DOCTYPE: &str = r#"<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">"#;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XmlError {
    pub offset: usize,
    pub reason: &'static str,
}

impl fmt::Display for XmlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "XML at byte {}: {}", self.offset, self.reason)
    }
}

impl core::error::Error for XmlError {}

pub fn parse(bytes: &[u8]) -> Result<Dictionary, XmlError> {
    let start_error = |reason| XmlError { offset: 0, reason };
    if bytes.len() > MAX_BYTES {
        return Err(start_error("configuration exceeds the byte limit"));
    }
    let mut text = str::from_utf8(bytes).map_err(|error| XmlError {
        offset: error.valid_up_to(),
        reason: "configuration must be UTF-8",
    })?;
    if let Some(without_bom) = text.strip_prefix('\u{feff}') {
        text = without_bom;
    }
    if let Some((offset, _)) = text.char_indices().find(|&(_, ch)| !xml_character(ch)) {
        return Err(XmlError {
            offset: bytes.len() - text.len() + offset,
            reason: "invalid XML character",
        });
    }
    let mut parser = Parser {
        rest: text,
        len: bytes.len(),
        nodes: 0,
    };
    if parser.take("<?xml") {
        parser.declaration()?;
    }
    parser.space()?;
    if parser.take(DOCTYPE) {
        parser.space()?;
    }
    parser.open_plist()?;
    let Value::Dictionary(dictionary) = parser.value(0)? else {
        return Err(parser.error("plist root must be a dictionary"));
    };
    parser.space()?;
    parser.expect_close("plist")?;
    parser.space()?;
    if !parser.rest.is_empty() {
        return Err(parser.error("extra input after plist"));
    }
    Ok(dictionary)
}

struct Parser<'a> {
    rest: &'a str,
    len: usize,
    nodes: usize,
}

impl<'a> Parser<'a> {
    fn error(&self, reason: &'static str) -> XmlError {
        XmlError {
            offset: self.len - self.rest.len(),
            reason,
        }
    }

    fn take(&mut self, text: &str) -> bool {
        if let Some(rest) = self.rest.strip_prefix(text) {
            self.rest = rest;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, text: &str) -> Result<(), XmlError> {
        self.take(text)
            .then_some(())
            .ok_or_else(|| self.error("unexpected tag"))
    }

    fn through(&mut self, delimiter: &str) -> Result<&'a str, XmlError> {
        let (content, rest) = self
            .rest
            .split_once(delimiter)
            .ok_or_else(|| self.error("unterminated XML construct"))?;
        self.rest = rest;
        Ok(content)
    }

    fn space(&mut self) -> Result<(), XmlError> {
        loop {
            self.rest = self.rest.trim_start_matches(xml_whitespace);
            if !self.take("<!--") {
                return Ok(());
            }
            let comment = self.through("-->")?;
            if comment.contains("--") || comment.ends_with('-') {
                return Err(self.error("invalid XML comment"));
            }
        }
    }

    fn whitespace(&mut self) -> bool {
        let trimmed = self.rest.trim_start_matches(xml_whitespace);
        let consumed = trimmed.len() < self.rest.len();
        self.rest = trimmed;
        consumed
    }

    fn attribute(&mut self, name: &str) -> Result<&'a str, XmlError> {
        self.expect(name)?;
        self.whitespace();
        self.expect("=")?;
        self.whitespace();
        if self.take("\"") {
            self.through("\"")
        } else if self.take("'") {
            self.through("'")
        } else {
            Err(self.error("expected a quoted attribute"))
        }
    }

    fn declaration(&mut self) -> Result<(), XmlError> {
        if !self.whitespace() || self.attribute("version")? != "1.0" {
            return Err(self.error("expected XML version 1.0"));
        }
        let mut separated = self.whitespace();
        if separated && self.rest.starts_with("encoding") {
            if !self.attribute("encoding")?.eq_ignore_ascii_case("UTF-8") {
                return Err(self.error("configuration must be UTF-8"));
            }
            separated = self.whitespace();
        }
        if separated && self.rest.starts_with("standalone") {
            if !matches!(self.attribute("standalone")?, "yes" | "no") {
                return Err(self.error("invalid standalone attribute"));
            }
            self.whitespace();
        }
        self.expect("?>")
    }

    fn open_plist(&mut self) -> Result<(), XmlError> {
        self.expect("<plist")?;
        if self.whitespace() && self.rest.starts_with("version") {
            if self.attribute("version")? != "1.0" {
                return Err(self.error("expected plist version 1.0"));
            }
            self.whitespace();
        }
        self.expect(">")
    }

    /// Consume the exact element name followed by XML whitespace and
    /// either a closing angle bracket or the empty-element delimiter.
    fn tag(&mut self, name: &str) -> Option<bool> {
        let rest = self.rest.strip_prefix('<')?.strip_prefix(name)?;
        let rest = rest.trim_start_matches(xml_whitespace);
        if let Some(rest) = rest.strip_prefix("/>") {
            self.rest = rest;
            Some(false)
        } else if let Some(rest) = rest.strip_prefix('>') {
            self.rest = rest;
            Some(true)
        } else {
            None
        }
    }

    fn close(&mut self, name: &str) -> bool {
        let Some(rest) = self
            .rest
            .strip_prefix("</")
            .and_then(|rest| rest.strip_prefix(name))
            .and_then(|rest| rest.trim_start_matches(xml_whitespace).strip_prefix('>'))
        else {
            return false;
        };
        self.rest = rest;
        true
    }

    fn expect_close(&mut self, name: &str) -> Result<(), XmlError> {
        self.close(name)
            .then_some(())
            .ok_or_else(|| self.error("unexpected tag"))
    }

    fn value(&mut self, depth: usize) -> Result<Value, XmlError> {
        self.space()?;
        self.nodes += 1;
        if depth > MAX_DEPTH || self.nodes > MAX_NODES {
            return Err(self.error("configuration exceeds the nesting or node limit"));
        }
        if let Some(open) = self.tag("dict") {
            return Ok(Value::Dictionary(self.dictionary(depth, open)?));
        }
        if let Some(open) = self.tag("array") {
            let mut values = Vec::new();
            if open {
                self.space()?;
                while !self.close("array") {
                    values.push(self.value(depth + 1)?);
                    self.space()?;
                }
            }
            return Ok(Value::Array(values));
        }
        if let Some(open) = self.tag("string") {
            return self.string(open, "string").map(Value::String);
        }
        if let Some(open) = self.tag("integer") {
            if !open {
                return Err(self.error("expected an integer"));
            }
            let text = self.string(true, "integer")?;
            let number = text
                .trim_matches(xml_whitespace)
                .parse::<i128>()
                .ok()
                .filter(|n| (i64::MIN as i128..=u64::MAX as i128).contains(n))
                .ok_or_else(|| self.error("expected a 64-bit decimal integer"))?;
            return Ok(Value::Integer(number));
        }
        if let Some(open) = self.tag("data") {
            let text = self.string(open, "data")?;
            return decode_base64(&text)
                .map(Value::Data)
                .ok_or_else(|| self.error("invalid base64 data"));
        }
        for (name, value) in [("true", true), ("false", false)] {
            if let Some(open) = self.tag(name) {
                if open {
                    self.space()?;
                    self.expect_close(name)?;
                }
                return Ok(Value::Boolean(value));
            }
        }
        Err(self.error("expected a supported plist element"))
    }

    fn dictionary(&mut self, depth: usize, open: bool) -> Result<Dictionary, XmlError> {
        let mut dictionary = Dictionary::new();
        if open {
            self.space()?;
            while !self.close("dict") {
                let open = self
                    .tag("key")
                    .ok_or_else(|| self.error("expected dictionary key"))?;
                let key = self.string(open, "key")?;
                let value = self.value(depth + 1)?;
                if dictionary.insert(key, value).is_some() {
                    return Err(self.error("duplicate dictionary key"));
                }
                self.space()?;
            }
        }
        Ok(dictionary)
    }

    fn string(&mut self, open: bool, close: &str) -> Result<String, XmlError> {
        if !open {
            return Ok(String::new());
        }
        let error = self.error("invalid XML character data");
        let end = self
            .rest
            .find('<')
            .ok_or_else(|| self.error("unterminated XML construct"))?;
        let text = &self.rest[..end];
        self.rest = &self.rest[end..];
        if !self.close(close) || text.contains("]]>") {
            return Err(error);
        }
        decode_entities(text).ok_or(error)
    }
}

fn xml_whitespace(ch: char) -> bool {
    matches!(ch, ' ' | '\t' | '\r' | '\n')
}

fn xml_character(ch: char) -> bool {
    matches!(ch, '\u{9}' | '\u{a}' | '\u{d}' | '\u{20}'..='\u{d7ff}'
        | '\u{e000}'..='\u{fffd}' | '\u{10000}'..='\u{10ffff}')
}

fn decode_entities(mut text: &str) -> Option<String> {
    let mut out = String::with_capacity(text.len());
    while let Some((before, entity)) = text.split_once('&') {
        push_literal(&mut out, before);
        let (name, rest) = entity.split_once(';')?;
        text = rest;
        let ch = match name {
            "amp" => '&',
            "lt" => '<',
            "gt" => '>',
            "apos" => '\'',
            "quot" => '"',
            _ => {
                let number = if let Some(hex) = name.strip_prefix("#x") {
                    if !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                        return None;
                    }
                    u32::from_str_radix(hex, 16).ok()?
                } else {
                    let decimal = name.strip_prefix('#')?;
                    if !decimal.bytes().all(|byte| byte.is_ascii_digit()) {
                        return None;
                    }
                    decimal.parse().ok()?
                };
                char::from_u32(number).filter(|&ch| xml_character(ch))?
            }
        };
        out.push(ch);
    }
    push_literal(&mut out, text);
    Some(out)
}

// XML 1.0 §2.11 normalizes literal CR and CRLF before character references
// are expanded. In particular, the numeric reference &#13; stays a CR.
fn push_literal(out: &mut String, mut text: &str) {
    while let Some((before, after)) = text.split_once('\r') {
        out.push_str(before);
        out.push('\n');
        text = after.strip_prefix('\n').unwrap_or(after);
    }
    out.push_str(text);
}

#[cfg(feature = "std")]
const BASE64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn decode_base64(text: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let mut quartet = [0; 4];
    let mut count = 0;
    let mut finished = false;
    for byte in text
        .bytes()
        .filter(|byte| !xml_whitespace(char::from(*byte)))
    {
        if finished {
            return None;
        }
        quartet[count] = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => 64,
            _ => return None,
        };
        count += 1;
        if count == quartet.len() {
            let [a, b, c, d] = quartet;
            if a == 64 || b == 64 {
                return None;
            }
            out.push((a << 2) | (b >> 4));
            match (c, d) {
                (64, 64) if b & 15 == 0 => finished = true,
                (64, _) => return None,
                (_, 64) if c & 3 != 0 => return None,
                (_, 64) => {
                    out.push((b << 4) | (c >> 2));
                    finished = true;
                }
                _ => {
                    out.push((b << 4) | (c >> 2));
                    out.push((c << 6) | d);
                }
            }
            count = 0;
        }
    }
    (count == 0).then_some(out)
}

/// Serialize a dictionary as UTF-8 XML. BTreeMap keeps writes deterministic.
#[cfg(feature = "std")]
pub fn to_xml(dictionary: &Dictionary) -> String {
    let mut out = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str(DOCTYPE);
    out.push_str("\n<plist version=\"1.0\">\n");
    write_dictionary(&mut out, dictionary, 0);
    out.push_str("</plist>\n");
    out
}

#[cfg(feature = "std")]
fn write_dictionary(out: &mut String, dictionary: &Dictionary, depth: usize) {
    indent(out, depth);
    out.push_str("<dict>\n");
    for (key, value) in dictionary {
        indent(out, depth + 1);
        out.push_str("<key>");
        escape(out, key);
        out.push_str("</key>\n");
        write_value(out, value, depth + 1);
    }
    indent(out, depth);
    out.push_str("</dict>\n");
}

#[cfg(feature = "std")]
fn write_value(out: &mut String, value: &Value, depth: usize) {
    use core::fmt::Write;
    if let Value::Dictionary(dictionary) = value {
        write_dictionary(out, dictionary, depth);
        return;
    }
    indent(out, depth);
    match value {
        Value::Dictionary(_) => unreachable!(),
        Value::Array(values) => {
            out.push_str("<array>\n");
            for value in values {
                write_value(out, value, depth + 1);
            }
            indent(out, depth);
            out.push_str("</array>\n");
        }
        Value::String(text) => {
            out.push_str("<string>");
            escape(out, text);
            out.push_str("</string>\n");
        }
        Value::Integer(number) => {
            let _ = writeln!(out, "<integer>{number}</integer>");
        }
        Value::Boolean(true) => out.push_str("<true/>\n"),
        Value::Boolean(false) => out.push_str("<false/>\n"),
        Value::Data(bytes) => {
            out.push_str("<data>");
            for chunk in bytes.chunks(3) {
                let a = chunk[0];
                let b = chunk.get(1).copied().unwrap_or(0);
                let c = chunk.get(2).copied().unwrap_or(0);
                out.push(char::from(BASE64[usize::from(a >> 2)]));
                out.push(char::from(BASE64[usize::from((a & 3) << 4 | b >> 4)]));
                out.push(if chunk.len() >= 2 {
                    char::from(BASE64[usize::from((b & 15) << 2 | c >> 6)])
                } else {
                    '='
                });
                out.push(if chunk.len() == 3 {
                    char::from(BASE64[usize::from(c & 63)])
                } else {
                    '='
                });
            }
            out.push_str("</data>\n");
        }
    }
}

#[cfg(feature = "std")]
fn indent(out: &mut String, depth: usize) {
    for _ in 0..depth {
        out.push_str("  ");
    }
}

#[cfg(feature = "std")]
fn escape(out: &mut String, text: &str) {
    for ch in text.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '\r' => out.push_str("&#13;"),
            _ => out.push(ch),
        }
    }
}
