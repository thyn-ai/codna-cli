export function fill(pdf: string): string {
  return `/usr/bin/pdftk ${pdf} fill_form output out.pdf`;
}
