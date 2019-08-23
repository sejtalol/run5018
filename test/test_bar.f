      program mytest
      implicit none

      integer i,j,k
      integer ios
      integer m_sp, n_sp, snp, irun, istep, ileap, iq
      integer nnsn,nnch
      parameter( m_sp =1000000)
      real sn(m_sp*3), snv(m_sp*3)
      real mass(m_sp), mang(m_sp), mpe(m_sp), mte(m_sp)
      integer nw, tmpn
      real tmptime
      integer flag_i(m_sp), flag_f(m_sp), cflag(m_sp)
      parameter( nw = 10000)
      real myrad(nw), myomega(nw), mykappa(nw), mykappaz(nw)
      integer step_interval, bar_r, iunit
      real tr(m_sp)
      character*20 filename
      parameter( step_interval = 8000 )
      parameter( bar_r = 3.2 )

      nnsn=35
      open (nnsn,file='../../run5018.snp',form='unformatted')

      read(nnsn)
      read(nnsn)
      read(nnsn)
      read(nnsn)

      nnch=36
      write(filename,"('bar_ptcls_',I4.4,'.dat')")step_interval
      print *, 'Filename = ', filename
      open(nnch,file=filename,status='unknown')

      do 
        read(nnsn,iostat=ios)irun,snp,istep,n_sp,tmptime
        if(istep .gt. 15000)exit
        read(nnsn)(sn(iq),iq=1,n_sp*3)
        read(nnsn)(snv(iq),iq=1,n_sp*3)
        read(nnsn)(mass(iq),iq=1,n_sp)
        read(nnsn)(mang(iq),iq=1,n_sp)
        read(nnsn)(mpe(iq),iq=1,n_sp)
        read(nnsn)(mte(iq),iq=1,n_sp)
        read(nnsn) tmpn
        do i = 1, tmpn, 1
            read(nnsn) myrad(i), myomega(i), mykappa(i), mykappaz(i)
        end do

        if(istep .eq. 7000)then
         do i = 1, n_sp, 1
          tr(i) = sqrt(sn(3*i-2)**2+sn(3*i-1)**2)
          if(tr(i) .lt. bar_r)then
            flag_i(i) = 1
            flag_f(i) = 1
            cflag(i) = 15000
c            print *, 'R = ',sqrt(sn(3*i-2)**2+sn(3*i-1)**2), 'is in bar'
          else
            flag_i(i) = 0
            flag_f(i) = 0
            cflag(i) = 7000
c            print *,'R = ',sqrt(sn(3*i-2)**2+sn(3*i-1)**2), 'not in bar'
          endif
         end do
        else if(istep .gt. 7000)then
         do i = 1, n_sp, 1
          tr(i) = sqrt(sn(3*i-2)**2+sn(3*i-1)**2)
          if(tr(i) .lt. bar_r .and. flag_f(i) == 1)then
                    flag_f(i) = 1
                else if(flag_f(i) == 1)then
                    flag_f(i) = 0
                    cflag(i) = istep
                else
                    flag_f(i) = 0
                end if
            enddo
       end if
      end do

      do i = 1, n_sp, 1
      write(nnch,10) i, flag_i(i), flag_f(i), cflag(i)
10      format(2X, I7, I3, I3, I7)
      end do
      end
