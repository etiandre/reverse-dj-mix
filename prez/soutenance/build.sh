#!/bin/sh

typst compile --root ../.. soutenance.typ && polylux2pdfpc --root ../.. soutenance.typ
